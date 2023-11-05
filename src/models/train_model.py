import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler
import numpy as np
import time
import math
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model'
    )
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    parser.add_argument('--hidden_size', dest="hidden_size", action="store", type=int, default=128)
    parser.add_argument('--batch_size', dest="batch_size", action="store", type=int, default=64)

    parser.add_argument('--input_path', dest="input_path", action="store",
                        default="..\\..\\data\\interim\\data_preprocessed.tsv")
    parser.add_argument('--output_path', dest="output_path", action="store",
                        default="..\\..\\models\\")
    args = parser.parse_args()
    if args.input_path.endswith(".tsv"):
        data = pd.read_csv(args.input_path, sep='\t')
    else:
        data = pd.read_csv(args.input_path)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class Lang:
        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS

        def addSentence(self, sentence):
            for word in sentence:
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1


    def readLangs(lang1, lang2):
        ref = reduced_data['reference'].apply(lambda x: eval(x))
        trn = reduced_data['translation'].apply(lambda x: eval(x))
        pairs = []
        for i in range(len(ref)):
            pairs.append((ref.iloc[i], trn.iloc[i]))
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        return input_lang, output_lang, pairs


    def prepareData(lang1, lang2):
        input_lang, output_lang, pairs = readLangs(lang1, lang2)
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, dropout_p=0.1):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, input):
            embedded = self.dropout(self.embedding(input))
            output, hidden = self.gru(embedded)
            return output, hidden


    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super(BahdanauAttention, self).__init__()
            self.Wa = nn.Linear(hidden_size, hidden_size)
            self.Ua = nn.Linear(hidden_size, hidden_size)
            self.Va = nn.Linear(hidden_size, 1)

        def forward(self, query, keys):
            scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
            scores = scores.squeeze(2).unsqueeze(1)

            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, keys)

            return context, weights


    class AttnDecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size, dropout_p=0.1):
            super(AttnDecoderRNN, self).__init__()
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.attention = BahdanauAttention(hidden_size)
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
            self.out = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
            batch_size = encoder_outputs.size(0)
            decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
            decoder_hidden = encoder_hidden
            decoder_outputs = []
            attentions = []

            for i in range(MAX_LENGTH):
                decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_outputs.append(decoder_output)
                attentions.append(attn_weights)

                if target_tensor is not None:
                    # Teacher forcing: Feed the target as the next input
                    decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()  # detach from history as input

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            attentions = torch.cat(attentions, dim=1)

            return decoder_outputs, decoder_hidden, attentions

        def forward_step(self, input, hidden, encoder_outputs):
            embedded = self.dropout(self.embedding(input))

            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            input_gru = torch.cat((embedded, context), dim=2)

            output, hidden = self.gru(input_gru, hidden)
            output = self.out(output)

            return output, hidden, attn_weights


    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence]


    def tensorFromSentence(lang, sentence):
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


    def tensorsFromPair(pair):
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)


    def get_dataloader(batch_size):
        input_lang, output_lang, pairs = prepareData('Ref', 'Tra')
        n = len(pairs)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                   torch.LongTensor(target_ids).to(device))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return input_lang, output_lang, train_dataloader


    def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, criterion):

        total_loss = 0
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


    def train_(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100):
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        for epoch in range(1, n_epochs + 1):
            loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_loss_avg))


    SOS_token = 0
    EOS_token = 1
    # Reduciing the size of the data
    data_ = data['reference'].apply(lambda x: len(eval(x)))
    data__ = data['translation'].apply(lambda x: len(eval(x)))
    reduced_data = data[data_ <= 127]
    reduced_data = reduced_data[data__ <= 127]
    reduced_data = reduced_data[reduced_data['ref_tox'] > reduced_data['trn_tox']]
    test_ratio = 0.1
    train_val, test = train_test_split(
        reduced_data, test_size=test_ratio, random_state=42)
    val_ratio = 0.2
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=42
    )
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    MAX_LENGTH = 128
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    train_(train_dataloader, encoder, decoder, args.epochs, print_every=5)
    n = len(os.listdir(args.output_path))
    torch.save(encoder.state_dict(), args.output_path + 'encoder_' + str(n//2) + '.pth')
    torch.save(decoder.state_dict(), args.output_path + 'decoder_' + str(n//2) + '.pth')
    print(f"Weights are saved at {args.output_path}")