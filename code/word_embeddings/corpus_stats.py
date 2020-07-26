import os
import csv

def analyze_corpus(dir_path):
    vocab = set()
    tokens = 0
    files = 0
    for filename in os.listdir(dir_path):
        with open(f'{dir_path}/{filename}') as csv_file:
            files += 1
            print(filename)
            r = csv.reader(csv_file, delimiter=' ')
            sents = [row for row in r]
            for line in sents:
                for word in line:
                    vocab.add(word)
                    tokens += 1
    return len(vocab), tokens, files

if __name__ == '__main__':
    print(analyze_corpus('sent_tokenized_corpus'))



