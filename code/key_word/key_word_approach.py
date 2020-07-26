import pandas as pd
import csv
import re



def score_segment_bigram(segment, concept):
    """
    Rank the segment using the key word search algorithm
    :param segment (list): a list of the tokens in the segment
    :param concept (list): list of the two tokens in the bigram
    :return: a numeric score of the number of occurences of the concept word
    """
    score = 0
    prev_word = ''
    concept[0] = concept[0].lower()
    concept[1] = concept[1].lower()
    for word in segment:
        if [prev_word.lower(), word.lower()] == concept:
            score += 1
        prev_word = word
    return score

def read_ranked_segments(path):
    rows = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]
    return rows[0]

def read_test_segments(path):
    """
    Read the segments for the csv file, and return in list from
    :param path: path to csv file
    :return:
    """
    segments = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        segments = [row for row in reader]
    return segments


def evaluate_concept_score_only(segments, concept, scoring_func=score_segment):
    """
    Score the list of segments for the given concept.
    :param segments: list of segments where seg id is seg[0] and segment tokens are seg[1:]
    :param concept: the str rep of the concept
    :return: list of dicts {id: seg_id-concept, score: score, rank: numeric rank}
    """
    rankings = []
    for segment in segments:
        seg_id = segment[0]
        seg_tokens = segment[1:]
        score = scoring_func(seg_tokens, concept)
        id = ''
        if type(concept) == list:
            id = f'{seg_id}-{" ".join([w.capitalize() for w in concept])}'
        else:
            id = f'{seg_id}-{concept.capitalize()}'
        rankings.append({'id': id, 'seg_id': seg_id, 'score': score})

    return rankings

def evaluate_concept(segments, concept, scoring_func=score_segment):
    """
    Score the list of segments for the given concept.
    :param segments: list of segments where seg id is seg[0] and segment tokens are seg[1:]
    :param concept: the str rep of the concept
    :return: list of dicts {id: seg_id-concept, score: score, rank: numeric rank}
    """
    rankings = []
    for segment in segments:
        seg_id = segment[0]
        seg_tokens = segment[1:]
        score = scoring_func(seg_tokens, concept)
        id = ''
        if type(concept) == list:
            id = f'{seg_id}-{" ".join([w.capitalize() for w in concept])}'
        else:
            id = f'{seg_id}-{concept.capitalize()}'
        rankings.append({'id': id, 'seg_id': seg_id, 'score': score})
    in_order = sorted(rankings, key=lambda ranking: ranking['score'], reverse=True)

    for i in range(len(in_order)):
        in_order[i]['rank'] = i
    return in_order

def filter_out_unranked(segments, ranked_segs=[]):
    """
    Given the list of segments filter out those segments whose ids are not in ranked_segs. id is segment[0]
    :param segments: list of segments
    :param ranked_segs: list of ranked segment ids
    :return: list of segments, ranked segments only
    """
    ranked_only = []
    excluded = 0
    for segment in segments:
        id = segment[0]
        if id in ranked_segs:
            ranked_only.append(segment)
        else:
            excluded += 1
    return ranked_only, excluded

def run_experiment_score_only(segments, concepts=[], ranked_segs=[], bigram_concepts=[]):
    """
    Run the kw algorithm for the given segments for the given concepts. Return results in a dataframe
    :param segments: list of segments where seg id is seg[0] and segment tokens are seg[1:]
    :param concepts: list of concepts represented as strs
    :return: dataframe index=seg_id-concept col=score
    """
    data = {'kw_relevance': []}
    index = []
    filtered_segs, excluded = filter_out_unranked(segments, ranked_segs=ranked_segs)
    for concept in concepts:
        rankings = evaluate_concept_score_only(filtered_segs, concept)
        for ranking in rankings:
            data['kw_relevance'].append(ranking['score'] > 0)
            index.append(ranking['id'])

    for concept in bigram_concepts:
        rankings = evaluate_concept_score_only(filtered_segs, concept, scoring_func=score_segment_bigram)
        for ranking in rankings:
            data['kw_relevance'].append(ranking['score'] > 0)
            index.append(ranking['id'])

    return pd.DataFrame(data=data, index=index)

def run_experiment(segments, concepts=[], ranked_segs=[], bigram_concepts=[]):
    """
    Run the kw algorithm for the given segments for the given concepts. Return results in a dataframe
    :param segments: list of segments where seg id is seg[0] and segment tokens are seg[1:] 
    :param concepts: list of concepts represented as strs
    :return: dataframe index=seg_id-concept col=score
    """
    data = {'kw_relevance': []}
    index = []
    filtered_segs, excluded = filter_out_unranked(segments, ranked_segs=ranked_segs)
    for concept in concepts:
        rankings = evaluate_concept(filtered_segs, concept)
        for ranking in rankings:
            data['kw_relevance'].append(ranking['rank'])
            index.append(ranking['id'])

    for concept in bigram_concepts:
        rankings = evaluate_concept(filtered_segs, concept, scoring_func=score_segment_bigram)
        for ranking in rankings:
            data['kw_relevance'].append(ranking['rank'])
            index.append(ranking['id'])

    return pd.DataFrame(data=data, index=index)


if __name__ == '__main__':
    SA = read_test_segments('../ricoeur/test_data/raw_segments/soi-meme-full-tokens.csv')
    SA_ranked = read_ranked_segments('../expert_judgements/ranked_segments/SA-ranked-segments.csv')

    SM = read_test_segments('../ricoeur/test_data/raw_segments/symb-du-mal-full-tokens.csv')
    SM_ranked = read_ranked_segments('../expert_judgements/ranked_segments/SM-ranked-segments.csv')

    #SA_df = run_experiment(SA, concepts=['justice', 'morale'], ranked_segs=SA_ranked, bigram_concepts=[['Sagesse', 'Pratique']])
    #SM_df = run_experiment(SM, concepts=['mythe', 'symbole', 'homme'], ranked_segs=SM_ranked)


    #SM_df.to_csv('SM-kw-results.csv')

    SM_df = run_experiment_score_only(SM, concepts=['mythe', 'homme', 'symbole'], ranked_segs=SM_ranked,)

    SM_df.to_csv('SM-kw-results-no-rank.csv')




