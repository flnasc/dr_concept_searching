import pandas as pd
if __name__ == '__main__':
    experts = pd.read_csv('../result_analysis/SM-judgements.csv', index_col=0)
    res = pd.read_csv('results_wmd_top_20.csv', index_col=0)
    print(experts)
    print(res)
    joined = res.join(experts, how='left', sort=False)
    joined.to_csv('wmd_top_20_w_raw_judgements.csv')