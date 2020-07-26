import pandas as pd
import gensim
import csv
import random
import spacy


def acs(concept, paragraph, model, min_length, stop_words=[]):
    """
    :param concept: str, the concept word for the concept
    :param paragraph: list, a list of the tokens in the paragraph
    :param model: gensim.models.Word2Vec, model containing word vectors
    :return: float, the distance between the concept and the paragraph
            as the sum of the cos distance between each word and the concept
            divided by the number of words.
    """
    d_sum = 0
    word_count = 0
    if len(paragraph) == 0:
        return d_sum
    if len(paragraph) <= min_length:
        return -1
    for word in paragraph:
        if word in model.wv and word not in stop_words:
            sim = model.wv.similarity(word, concept)
            d_sum += sim
            word_count += 1

    return d_sum/word_count


def load_segments(path):
    data = {'segment': []}
    index = []
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        for row in r:
            data['segment'].append(row[1:])
            index.append(row[0])

    return pd.DataFrame(data=data, index=index)

def rank_using_acs(gensim_model, segments, concept, stop_words=[]):
    distances = []
    for indx, row in segments.iterrows():
        acs_score = acs(concept, row['segment'], gensim_model, 30, stop_words=stop_words)
        distances.append({'id': indx, 'acs_score': acs_score})

    srt_distances = sorted(distances, key=lambda x: x['acs_score'], reverse=True)
    print(srt_distances[:10])
    rankings = [{'id': seg['id'], 'rank': i} for i, seg in enumerate(srt_distances)]
    print(rankings[:10])

    data = {'rank':[]}
    index = []
    for ranking in rankings:
        data['rank'].append(ranking['rank'])
        index.append(f'{ranking["id"]}-{concept.capitalize()}')

    return pd.DataFrame(data=data, index=index)



if __name__ == '__main__':

    # load spacy for stop words
    nlp = spacy.load("fr_core_news_sm")
    stop_words = nlp.Defaults.stop_words
    # load segments and model
    print('loading segments and model')
    segments = load_segments('../test_set/symb-du-mal-full-tokens-only-ranked-comma.csv')
    gensim_model = gensim.models.Word2Vec.load('../word_embeddings/model/word2vec_french_w_pretrained_jun_2020.model')
    print(segments.head())

    # normalize word vectors (good for WMD) may take some time
    print('normalizing word embeddings')
    gensim_model.init_sims(replace=True)

    print('ranking')
    morale = rank_using_acs(gensim_model, segments, 'homme', stop_words=stop_words)
    morale.to_csv('homme-acs.csv')