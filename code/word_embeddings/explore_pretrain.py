import gensim

if __name__ == '__main__':
    print('loading pretrained model')
    model = gensim.models.KeyedVectors.load_word2vec_format('43/model.txt', binary=False, unicode_errors='replace')
    model = gensim.models.KeyedVectors.load_word2vec_format('43/model.txt', binary=False, unicode_errors='replace')
    while True:
        try:
            word = input('Word: ')
            print(model.most_similar(word, topn=20))
        except KeyError:
            continue
        except KeyboardInterrupt:
            quit(0)


