import pandas as pd
from sklearn.metrics import precision_score

def get_precision(experts, strategy, top_n):
    agg_df = strategy.join(experts)
    print(agg_df.to_string())
    agg_df.to_csv('result.csv')
    precision_vals = []
    for n in top_n:
        pred = [i < n for i in agg_df['rank']]
        actual = [i == 'Relevant' for i in agg_df['relevance']]
        prec = precision_score(actual, pred)
        precision_vals.append({'top_n': n, 'prec': prec})
    return precision_vals
if __name__ == '__main__':
    acs = pd.read_csv('symbole-acs.csv', index_col=0)
    kw = pd.read_csv('symbole-kw.csv', index_col=0)
    wmd = pd.read_csv('Symbole-gensim-wmd.csv', index_col=0)
    experts = pd.read_csv('SM-relevance-majority.csv', index_col=0)
    acs_vals = get_precision(experts, acs, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    kw_vals = get_precision(experts, kw, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    wmd_vals = get_precision(experts, wmd, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    data = []
    cols = ['n','kw', 'acs','wmd']
    for i in range(len(acs_vals)):
        row = [i+1]
        row.append(kw_vals[i]['prec'])
        row.append(acs_vals[i]['prec'])
        row.append(wmd_vals[i]['prec'])
        data.append(row)
    print(data)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv('symbole-results-unnormalized.csv')



