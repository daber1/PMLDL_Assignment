import argparse
import torch
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make a prediction'
    )
    parser.add_argument('--encoder', dest="encoder", action="store", default="..\\..\\models\\encoder.pth")
    parser.add_argument('--decoder', dest="decoder", action="store", default="..\\..\\models\\decoder.pth")
    parser.add_argument('--input_path', dest="input_path", action="store",
                        default="..\\..\\data\\interim\\data_preprocessed.tsv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.input_path.endswith(".tsv"):
        data = pd.read_csv(args.input_path, sep='\t')
    else:
        data = pd.read_csv(args.input_path)

    print("Please, wait for the model to be loaded...")
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

    def evaluate(encoder, decoder, sentence, input_lang, output_lang):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
        return decoded_words, decoder_attn
    SOS_token = 0
    EOS_token = 1
    MAX_LENGTH = 128
    encoder_state = torch.load(args.encoder)
    decoder_state = torch.load(args.decoder)
    hidden_size = encoder_state['embedding.weight'].shape
    data_ = data['reference'].apply(lambda x: len(eval(x)))
    data__ = data['translation'].apply(lambda x: len(eval(x)))
    reduced_data = data[data_ <= 127]
    reduced_data = reduced_data[data__ <= 127]
    reduced_data = reduced_data[reduced_data['ref_tox'] > reduced_data['trn_tox']]
    input_lang, output_lang, _ = prepareData('Ref', 'Tra')
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    sentence = input("Enter a sentence")

    from nltk.tokenize import word_tokenize


    def tokenize_text(text):
        return word_tokenize(text)


    import re


    def lower_text(text):
        return text.lower()


    def remove_numbers(text):
        text_nonum = re.sub(r'\d+', ' ', text)
        return text_nonum


    def remove_punc(text):
        text_nopunc = re.sub(r'[^a-z|\s]', ' ', text)
        return text_nopunc


    def remove_multi_spaces(text):
        text_no_doublespaces = re.sub('\s+', ' ', text).strip()
        return text_no_doublespaces
    def preprocess(text):
        _lowered = lower_text(text)
        _without_numbers = remove_numbers(_lowered)
        _without_punct = remove_punc(_without_numbers)
        _single_spaced = remove_multi_spaces(_without_punct)
        _tokenized = tokenize_text(_single_spaced)

        return _tokenized

    sentence = preprocess(sentence)
    _, output = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    print("The output of the model: " + ' '.join(output))