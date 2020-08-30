import pandas as pd
import csv
from sentence_transformers import SentenceTransformer, util

def load_segments(path):
    data = {'segment': []}
    index = []
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        for row in r:
            data['segment'].append(row[1:])
            index.append(row[0])

    return pd.DataFrame(data=data, index=index)

def getBERTSimilarity(model, definition, segment):
    def_embed = model.encode(definition, convert_to_tensor=True)
    segment_embed = model.encode(segment, convert_to_tensor=True)
    return util.pytorch_cos_sim(def_embed, segment_embed).numpy()[0][0]




def rank_using_gensim_wmd(model, segments, definition_id, concept):
    distances = []
    definition_vec = segments.loc[definition_id]['segment']
    for indx, row in segments.iterrows():
        if indx != definition_id and not indx.startswith('def'):
            distance = getBERTSimilarity(model, ' '.join(definition_vec), ' '.join(row['segment']))
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
    segments = load_segments('../test_set/symb-du-mal-full-tokens-only-ranked-stop-words-wmd.csv')
    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    print(segments.head())


    print('ranking')
    morale = rank_using_gensim_wmd(model, segments, 'def-homme', 'Homme')
    morale.to_csv('Homme-bert.csv')