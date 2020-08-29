import pandas as pd
import csv
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

def load_segments(path):
    data = {'segment': []}
    index = []
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        for row in r:
            data['segment'].append(row[1:])
            index.append(row[0])

    return pd.DataFrame(data=data, index=index)

def getUSESimilarity(google_use, definition, segment):
    def_embed = google_use(definition)
    segment_embed = google_use(segment)
    return np.inner(def_embed, segment_embed)[0][0]




def rank_using_gensim_wmd(google_use, segments, definition_id, concept):
    distances = []
    definition_vec = segments.loc[definition_id]['segment']
    for indx, row in segments.iterrows():
        if indx != definition_id and not indx.startswith('def'):
            distance = getUSESimilarity(google_use, ' '.join(definition_vec), ' '.join(row['segment']))
            distances.append({'id': indx, 'distance': distance})

    srt_distances = sorted(distances, key=lambda x: x['distance'], reverse=True)
    print(srt_distances[:10])
    rankings = [{'id': seg['id'], 'rank': i} for i, seg in enumerate(srt_distances)]
    print(rankings[:10])

    data = {'rank':[]}
    index = []
    for ranking in rankings:
        data['rank'].append(ranking['rank'])
        index.append(f'{ranking["id"]}-{concept}')

    return pd.DataFrame(data=data, index=index)



if __name__ == '__main__':

    # load segments and model
    print('loading segments and model')
    segments = load_segments('test_set/symb-du-mal-full-tokens-only-ranked-stop-words-wmd.csv')
    google_use = hub.load("universal-sentence-encoder-multilingual-large_3")
    print(segments.head())


    print('ranking')
    morale = rank_using_gensim_wmd(google_use, segments, 'def-mythe', 'Mythe')
    morale.to_csv('Mythe-use.csv')