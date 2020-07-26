# from tutorial https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XYlUnpNKh0s
from gensim.utils import simple_preprocess
import gensim
import logging
import spacy
import csv
import os
sentences = None

def sentence_iter(dir_path):
    count = 0
    for file in os.listdir(dir_path):
        with open(f'{dir_path}/{file}', 'r') as csvfile:
            print(f'{dir_path}/{file}')
            reader = csv.reader(csvfile)
            for row in reader:
                count += len(row)
                print(count)
                yield row

model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=1, workers=10, iter=200)
if __name__ == '__main__':
    sentences = gensim.models.word2vec.PathLineSentences('../preprocessing/english/tokenized_corpus')
    print('training')
    model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=1, workers=10, iter=200)
    print(model.wv.vocab)
    model.save("word2vec_english_1.model")