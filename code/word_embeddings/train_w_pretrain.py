# from tutorial https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XYlUnpNKh0s
import gensim
import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    sentences = gensim.models.word2vec.PathLineSentences('sent_tokenized_corpus')
    model = gensim.models.Word2Vec.load('model/word2vec_french_pretrained.model')
    print('training model')
    start = time.time()
    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=100)
    end = time.time()
    print(f"{(end - start) / 60} minutes")
    model.save('word2vec_french_w_pretrained_4_iter_100.model')
    pass
