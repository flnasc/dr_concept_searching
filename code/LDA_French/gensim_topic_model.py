"""
   Author: Dylan Hayton-Ruffner
   Description: Runs lda on the given corpus, prints out resulting topics and queries every concept from the concept file.

            Query: If the concept-word exsists in the top 4 words of a topic, all the paragraphs associated with that topic and have
            the concept word are returned

            After each successful query, the results are formated into an excel file and written to the results folder.

   Status: Finished
   ToDo: N/A

   NOTES: Concept path and results path are hard-coded


"""
TOPIC_PRESSENCE_THRESHOLD = 0.3
REGEX_PATTERN = u'(?u)\\b\\w\\w\\w\\w+\\b'
MIN_WORD_COUNT = 10
NUM_TOPICS = 7
TOP_N_SEGS = 10
TOP_N_WORDS = 0
MIN_DF = 0.00
MAX_DF = 1.00
FILETYPE = 'xml'
CONCEPTS_PATH = "../../data/concepts.txt"
import pyLDAvis
import pyLDAvis.gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import numpy as np
import sys
import random
from operator import itemgetter
from nltk import word_tokenize
# from elbow_criteria import threshold
# from elbow_criteria import limit_by_threshold
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import gensim
import matplotlib.pyplot as plt
import csv


############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECITON-----")
    corpus = load_from_csv("../../data/lemmatized_segments/symb-du-mal-full-lemma.csv")

    # Create CountVectorizer to get Document-Term matrix

    stop_words = load_stop_words("../../data/stopwords-fr.txt")
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer)
    proc_corpus_text_only = [seg.split() for seg in proc_corpus_text_only]
    proc_stop_words = []

    for i in range(len(proc_corpus_text_only)):
        proc_stop_words.append([])
        for j in range(len(proc_corpus_text_only[i])):
            if proc_corpus_text_only[i][j] not in stop_words and len(proc_corpus_text_only[i][j]) >= 3:
                proc_stop_words[i].append(proc_corpus_text_only[i][j])

    # train vectorizer on corpus

    id2word = Dictionary(proc_stop_words)
    corp = [id2word.doc2bow(text) for text in proc_stop_words]

    # print("Number of Features: " + str(len(feature_names)))

    # initialize model
    path_to_mallet_binary = "Mallet/bin/mallet"

    mallet_model = LdaMallet(path_to_mallet_binary, corpus=corp, num_topics=18, id2word=id2word, optimize_interval=20,
                             random_seed=4, iterations=1000)
    return 0


############################# LOAD DATA #############################
def load_from_csv(path):
    """
    Loads all the segments from a csvfile.
    :param path: string, path to csvfile
    :return: list, a list of all the segments
    """
    segs = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            segs.append(row)
    return segs


def remove_short_segs(corpus, vectorizer):
    """
    Remove the short segments from the corpus i.e. less than min word count.
    :param corpus: list, a list of all text segments
    :param vectorizer: CountVectorizer object, built for french
    :return: proc_corpus, a list of all text segments with # of words > min word count
    """

    proc_corpus = []
    proc_corpus_text_only = []
    for seg in corpus:
        id = seg[0]
        text = seg[1]
        vec = vectorizer.fit_transform([text])
        if vec.shape[1] > MIN_WORD_COUNT:
            proc_corpus.append([id, text])
            proc_corpus_text_only.append(text)

    return proc_corpus, proc_corpus_text_only


def load_stop_words(path):
    """
    Loads the stop words from txt file
    :param path: string, path to text file
    :return: list, list of stop words
    """
    stop_words = []
    with open(path) as txtfile:
        for line in txtfile:
            stop_words.append(line.strip().lower())
    return stop_words


def parse_concepts(concepts_raw):
    lines = concepts_raw.split('\n')
    concepts = []
    for line in lines:
        if len(line.split()) > 0:
            concepts.append(line.split())
    return concepts


def load_corpus_txt(c_size):
    # works with a raw text file
    filepath = input("Filepath to corpus: ")
    print("LOADING FILE: " + filepath)
    doc_string = load_document(filepath)
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)
    preproc = vectorizer.build_analyzer()
    proc_text = preproc(doc_string)

    count = -1
    corpus_i = 0
    corpus = []
    n = c_size
    final = [proc_text[i * n:(i + 1) * n] for i in range((len(proc_text) + n - 1) // n)]

    return final

############################# Test Elbow Algorithm #############################


def run_elbow(model, feature_names):
    """Prints the topic information. Takes the sklearn.decomposition.LatentDiricheltAllocation lda model,
    the names of all the features, the number of words to be printined per topic, a list holding the freq
    of each topic in the corpus"""
    print("Elbow Limited Topics:")
    message_list = []

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % (topic_idx)

        # get the names of the features in sorted order -> argsort() return sorted indicies
        list_feat = [feature_names[i]
                     for i in topic.argsort()[::-1]]  # [::-1] reverses list

        # get the frequencis of the top words (limited by the threshold function)
        feat_freq = sorted(topic, reverse=True)
        cutoff = threshold(sorted(topic, reverse=True))
        limited_freq = limit_by_threshold(feat_freq, cutoff)

        for j in range(len(limited_freq)):
            message += "%s: %s, " % (str(list_feat[j]), str(limited_freq[j]))

        message_list.append(message)
        print(message)
    print()

    return message_list



if __name__ == "__main__":
    main()
