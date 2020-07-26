import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_concepts(filepath):
    return open(filepath).readlines()
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    concepts = [word.strip() for word in get_concepts('concepts.txt')]

    for word in model.wv.vocab:
        if word in concepts:
            tokens.append(model[word])
            labels.append(word)
    print(len(tokens))
    tokens = tokens[:1000]
    labels = labels[:1000]
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':
    model_french = gensim.models.Word2Vec.load('../models/word2vec_french_1.model')
    model_english = gensim.models.Word2Vec.load('../models/model2.model')
    print(model_french.most_similar('soi'))
    print(model_english.most_similar('self'))



