import logging
import csv
import spacy
import time
import os
from multiprocessing import Pool
from spacy.tokenizer import Tokenizer


def valid(token):
    return not token.is_punct and not token.is_space

def get_text_units(text, max_text_unit_length):
    return [text[i:min(len(text),i+max_text_unit_length)] for i in range(0, len(text), max_text_unit_length)]

def tokenize(nlp, raw_text, name, max_text_unit_length=1000000):
    s = time.time()

    # split text into units to feed to spacy
    print('Breaking text into processable units')
    text_units = get_text_units(raw_text, max_text_unit_length)

    # run the spacy model on the text, this is to tokenize by sentence
    total = len(text_units)

    for i, unit in enumerate(text_units):
        processed = nlp(unit)

        # get tokens from each sentence (sent is a span object)
        tk_sents = [[token.string.lower().strip() for token in sent if valid(token)] for sent in processed.sents]
        with open(f'{name}.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerows(tk_sents)
    e = time.time()
    print(f'took {e - s} seconds')

def tokenize_corpus(dir_path):
    nlp = spacy.load("fr_core_news_sm")
    for file in os.listdir(dir_path):
        print(file)
        raw_text = open(f'{dir_path}/{file}', 'r').read()
        print(f'Tokenizing {file}')
        tokenize(nlp, raw_text, file)

if __name__ == '__main__':
    tokenize_corpus('../corpus_in_french_jun_2020')