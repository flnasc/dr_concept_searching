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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_validate
import re
import numpy as np
import sys
import random
from operator import itemgetter
from nltk import word_tokenize
# from elbow_criteria import threshold
# from elbow_criteria import limit_by_threshold
import matplotlib.pyplot as plt
import csv



############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECITON-----")
    corpus = load_from_csv("../../data/lemmatized_segments/soi-meme-full-lemma.csv")

    # Create CountVectorizer to get Document-Term matrix
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    vectorizer.stop_words = load_stop_words("../../data/stopwords-fr.txt")

    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer)

    # train vectorizer on corpus
    dt_matrix = vectorizer.fit_transform(proc_corpus_text_only)

    feature_names = vectorizer.get_feature_names()
    # print("Number of Features: " + str(len(feature_names)))


    # initialize model
    print("initialize model")
    scores = []
    for i in range(1, 10):
        print("____Running_" + str(i) + "_Topics___")
        lda = LatentDirichletAllocation(n_components=i, max_iter=400,
                                        learning_method='batch', random_state=55, evaluate_every=5)

        info = cross_validate(lda, dt_matrix, scoring=perplexity_score, cv=10)
        scores.append(info["test_score"])
    for score in scores:
        print(list(score))
    return 0

############################# SCORING #############################


def perplexity_score(estimator, X, y=None):
    if y is not None:
        print("scoring function recieved target data")
        quit(1)
    return estimator.perplexity(X)

def harmonic_means_score(estimator, X, y=None):
    if y is not None:
        print("scoring function recieved target data")
        quit(1)
    print(X.type)
    return 1.0

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




if __name__ == "__main__":
    main()
