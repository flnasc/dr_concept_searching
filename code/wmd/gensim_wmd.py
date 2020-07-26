import pandas as pd
import gensim
import csv

def load_segments(path):
    data = {'segment': []}
    index = []
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        for row in r:
            data['segment'].append(row[1:])
            index.append(row[0])

    return pd.DataFrame(data=data, index=index)

def rank_using_gensim_wmd(gensim_model, segments, definition_id, concept):
    distances = []
    definition_vec = segments.loc[definition_id]['segment']
    for indx, row in segments.iterrows():
        if indx != definition_id and not indx.startswith('def'):
            distance = gensim_model.wmdistance(definition_vec, row['segment'])
            distances.append({'id': indx, 'distance': distance})

    srt_distances = sorted(distances, key=lambda x: x['distance'])
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
    segments = load_segments('test_set/symb-du-mal-full-tokens-only-ranked-stop-words.csv')
    gensim_model = gensim.models.Word2Vec.load('../word_embeddings/model/word2vec_french_w_pretrained_jun_2020.model')
    print(segments.head())

    # normalize word vectors (good for WMD) may take some time
    print('normalizing word embeddings')
    gensim_model.init_sims(replace=True)

    print('ranking')
    morale = rank_using_gensim_wmd(gensim_model, segments, 'def-homme', 'Homme')
    morale.to_csv('Homme-gensim-wmd.csv')