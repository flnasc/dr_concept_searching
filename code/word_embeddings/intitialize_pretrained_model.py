# from tutorial https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XYlUnpNKh0s
import gensim
import time

if __name__ == '__main__':
    print('loading sentences')
    sentences = gensim.models.word2vec.PathLineSentences('sent_tokenized_corpus')
    print('loading model')
    model = gensim.models.Word2Vec(size=100, window=5, min_count=1, workers=4, iter=200)
    print('building vocab sentences')
    start = time.time()
    model.build_vocab(sentences=sentences)
    end = time.time()
    print(f"{(end - start) / 60} minutes")
    print('intersecting pretrained vectors')
    start = time.time()
    model.intersect_word2vec_format('43/model.txt', lockf=1.0, binary=False, unicode_errors='replace')
    end = time.time()
    print(f"{(end - start) / 60} minutes")
    model.save('word2vec_french_pretrained.model')
    pass
