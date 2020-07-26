import pandas as pd
import gensim
import csv
import random


def kw_score(concept, segment):
    """
    Rank the segment using the key word search algorithm
    :param segment (list): a list of the tokens in the segment
    :param concept (str): the concept
    :return: a numeric score of the number of occurences of the concept word normalized by document length
    """
    score = 0
    for word in segment:
        if concept.lower() == word.lower():
            score += 1
    return score/len(segment)


def load_segments(path):
    data = {'segment': []}
    index = []
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        for row in r:
            data['segment'].append(row[1:])
            index.append(row[0])

    return pd.DataFrame(data=data, index=index)

def rank_using_kw(segments, concept):
    distances = []
    for indx, row in segments.iterrows():
        acs_score = kw_score(concept, row['segment'])
        distances.append({'id': indx, 'kw_score': acs_score})

    srt_distances = sorted(distances, key=lambda x: x['kw_score'], reverse=True)
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

    # load segments and model
    print('loading segments and model')
    segments = load_segments('../test_set/symb-du-mal-full-tokens-only-ranked-stop-words-acs.csv')
    print(segments.head())

    print('ranking')
    morale = rank_using_kw(segments, 'symbole')
    morale.to_csv('symbole-kw.csv')