import gensim
import numpy

if __name__ == '__main__':
    wmd2 = 0.004000000000000115
    t0 = 0.54
    print('loading pretrained model')
    model = gensim.models.Word2Vec.load("word2vec_french_w_pretrained_jun_2020.model")

    while True:
        try:
            word = input('Word: ')
            print(model.most_similar(word, topn=20))
        except KeyError:
            continue
        except KeyboardInterrupt:
            quit(0)