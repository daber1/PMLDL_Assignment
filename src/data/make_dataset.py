import requests
import zipfile
import io
import os
import pandas as pd
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Make a dataset'
    )
    link = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
    parser.add_argument('--data_link', dest="link", action="store", default=link)
    parser.add_argument('--output_path', dest="output_path", action="store", default="..\..\data\interim\\data_preprocessed.tsv")
    args = parser.parse_args()
    print("The dataset is downloading...")
    r = requests.get(args.link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    filename = z.namelist()[0]

    print("The dataset is downloaded.")
    if filename.endswith('.tsv'):
        data = pd.read_csv(filename, sep='\t')
    else:
        data = pd.read_csv(filename)
    data.drop(columns=["Unnamed: 0"], inplace=True)

    # Preprocessing

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

    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    def tokenize_text(text):
        return word_tokenize(text)


    def preprocess(text):
        _lowered = lower_text(text)
        _without_numbers = remove_numbers(_lowered)
        _without_punct = remove_punc(_without_numbers)
        _single_spaced = remove_multi_spaces(_without_punct)
        _tokenized = tokenize_text(_single_spaced)

        return _tokenized

    data['reference'] = data['reference'].apply(preprocess)
    data['translation'] = data['translation'].apply(preprocess)
    data = data[data['ref_tox']>data['trn_tox']]
    data.to_csv(args.output_path, sep='\t', index=False)
    print("Dataset is downloaded and preprocessed.")