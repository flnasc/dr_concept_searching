import pandas
from operator import itemgetter
import csv


to_numeric = {'Relates to': 3, 'Defines': 4, 'Sub concept': 2, 'Not related': 1}

def write(data, name):
    with open(name, 'w+') as csv_file:
        w = csv.writer(csv_file)
        for r in data:
            w.writerow(r)

def rank(tags):
    score = 0
    for tag in tags:
        score += to_numeric[tag]
    return score
def experts_rank(df):
    mythe = []
    homme = []
    symbole = []
    for index, row in df.iterrows():
        if 'Mythe' in index:
            mythe.append([index]+list(row)+[rank(list(row))])
        elif 'Homme' in index:
            homme.append([index]+list(row)+[rank(list(row))])
        elif 'Symbole' in index:
            symbole.append([index]+list(row)+[rank(list(row))])
    mythe = sorted(mythe, key=itemgetter(5), reverse=True)
    symbole = sorted(symbole, key=itemgetter(5), reverse=True)
    homme = sorted(homme, key=itemgetter(5), reverse=True)
    write(mythe, 'mythe_ex_rank.csv')
    write(symbole, 'symbole_ex_rank.csv')
    write(homme, 'homme_ex_rank.csv')

if __name__ == '__main__':
    df = pandas.read_csv('SM-judgements.csv', index_col=0)
    experts_rank(df)
