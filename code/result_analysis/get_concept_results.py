import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

def to_true_false(judgements, mapping={'Irrelevant': False, 'Relevant': True}):
    """
    Takes the array of relevance tags and turns them into True (relevant) False (irrelevant
    :param judgments:
    :param mapping
    :return: array of boolean values
    """
    return [mapping[judgement] for judgement in judgements]

def apply_top_n_relevance(rankings, top_n):
    """
    Given a list of numeric ranks, return a list of true/false relevance judgements
    using top_n thresholding
    :param rankings (list): list of numbers corresponding to the ranks for segments
    :return: list of booleans corresponding to segment relevance
    """
    judgements = []
    for rank in rankings:
        if rank is None or np.isnan(rank):
            judgements.append(False)
        elif type(rank) == bool:
            judgements.append(rank)
        else:
            judgements.append((rank < top_n))
    return judgements


def filter_for_concept(df, concept):
    """
    Filter the aggregated dataframe for all segment-concept pairs matching concept.
    Use to get all info on 1 concept for the book.
    :param df: aggregated dataframe [segment-concept, experts, trial_0, trial_1...]
    :param concept: a string representation of the concept
    :return: a new dataframe with only the segment concept pairs matching concept
    """
    if concept == 'Sagesse Pratique':
        return df[df.index.str.endswith(f'-{concept}')]
    else:
        return df[df.index.str.endswith(f'-{concept.capitalize()}')]

def get_precision_recall(y_true, y_pred):
    """
    Return precision and recall given y_true and y_pred
    :param y_true: list of ground truth judgements
    :param y_pred: list of algorithm pred
    :return: (precision, recall)
    """
    return (precision_score(y_true, y_pred), recall_score(y_true, y_pred))


def get_trial_results_df(expert_judgements, rankings, top_n_vals=[5,10,15,20]):
    """
    Produces a dataframe of precision and recall for the algorithm at each top_n value
    :param expert_judgements: expert relevance judgments for segments
    :param rankings: algorithm rankings for segments
    :param top_n_vals: top_n vals to use
    :return: dataframe
    """
    data = {'precision': [], 'recall': []}
    index = top_n_vals

    for val in top_n_vals:
        alg_relevance = apply_top_n_relevance(rankings, val)
        precision, recall = get_precision_recall(expert_judgements, alg_relevance)
        data['precision'].append(precision)
        data['recall'].append(recall)

    return pd.DataFrame(data=data, index=index)


def analyze_concept(agg_df, concept, top_n_vals=[5,10,15,20], column_regex='filtered_results/'):
    """
    :param agg_df: df with all info for one book
    :param concept (str): concept of interest
    :param top_n_vals: vals of top_n results from algorithm to deem relevant
    :return:
    """
    concept_df = filter_for_concept(agg_df, concept)
    trial_only = concept_df.filter(regex=column_regex, axis=1)
    experts = concept_df['relevance']
    experts_tf = to_true_false(experts)
    for col in trial_only.columns:
        rankings = trial_only[col]
        print(rankings)
        trial_results = get_trial_results_df(experts_tf, rankings, top_n_vals)
        col_name = col.replace('/', '-')
        trial_results.to_csv(f'{concept}-{col_name}.csv')


if __name__ == '__main__':
    # agg_df = pd.read_csv('agg_data/SA-agg-data-alg.csv', index_col=0)
    # analyze_concept(agg_df, 'justice')
    #
    # agg_df = pd.read_csv('agg_data/SA-agg-data-kw.csv', index_col=0)
    # analyze_concept(agg_df, 'justice', column_regex='kw_relevance')

    agg_df = pd.read_csv('agg_data/SA-agg-data-kw-no-rank.csv', index_col=0)
    analyze_concept(agg_df, 'Morale', column_regex='kw_relevance')
    analyze_concept(agg_df, 'Justice', column_regex='kw_relevance')
    analyze_concept(agg_df, 'Sagesse Pratique', column_regex='kw_relevance')



