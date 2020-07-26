import spacy
import csv
import os




def remove_stop_words(lines, stop_words):
    """
    Remove stop words from segments
    :param lines: list of segments [id, word, word..]
    :param stop_words: list of stop words (from spacy)
    :return: list of segments with stop_words removed
    """
    filtered_lines = []
    for line in lines:
        if len(line) < 2:
            print('empty line')
            filtered_lines.append(line)
            continue

        id = [line[0]]
        tokens = line[1:]
        filtered_tokens = [token for token in tokens if token not in stop_words]
        if '404e' in filtered_tokens:
            print(filtered_tokens)
        filtered_line = id + filtered_tokens
        filtered_lines.append(filtered_line)
    return filtered_lines

def process_file(path, stop_words):
    with open(path) as csv_file:
        r = csv.reader(csv_file, delimiter=' ')
        with open(f'{os.path.splitext(path)[0]}-stop-words.csv', 'w+') as new_csv_file:
            w = csv.writer(new_csv_file)
            lines = [line for line in r]
            filtered_lines = remove_stop_words(lines, stop_words)
            w.writerows(filtered_lines)

def load_unknown_words(path):
    with open(path) as csv_file:
        r = csv.reader(csv_file)
        rows = [row for row in r]
        return rows[0]
if __name__ == '__main__':
    nlp = spacy.load("fr_core_news_sm")
    stop_words = nlp.Defaults.stop_words
    #print('404e' in stop_words)
    process_file('symb-du-mal-full-tokens-only-ranked.csv', stop_words)





