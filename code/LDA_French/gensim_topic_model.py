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

import matplotlib.pyplot as plt
import csv


############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECITON-----")
    corpus = load_from_csv("../../data/lemmatized_segments/soi-meme-full-lemma.csv")

    # Create CountVectorizer to get Document-Term matrix
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    stop_words = load_stop_words("../../data/stopwords-fr.txt")

    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer)
    proc_corpus_text_only = [seg.split() for seg in proc_corpus_text_only]
    proc_stop_words = []
    for i in range(len(proc_corpus_text_only)):
        proc_stop_words.append([])
        for j in range(len(proc_corpus_text_only[i])):
            if proc_corpus_text_only[i][j] not in stop_words and len(proc_corpus_text_only[i][j]) >= 3:
                proc_stop_words[i].append(proc_corpus_text_only[i][j])
    for val in proc_stop_words:
        print(val)

    quit()
    # train vectorizer on corpus

    feature_names = vectorizer.get_feature_names()
    id2word = Dictionary(proc_stop_words)
    corp = [id2word.doc2bow(text) for text in proc_stop_words]

    # print("Number of Features: " + str(len(feature_names)))

    # initialize model
    # print("initialize model")
    path_to_mallet_binary = "Mallet/bin/mallet"
    res = []
    res_num = []
    for i in range(1,40):
        model = LdaMallet(path_to_mallet_binary, corpus=corp, num_topics=i, id2word=id2word)
        coherence_model_lda = CoherenceModel(model=model, texts=proc_stop_words, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        res.append("Topics: "+ str(i)+ " Coherence: "+ str(coherence_lda))
        res_num.append(coherence_lda)

    plt.plot(res_num)
    plt.show()
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


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


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


############################# Top-N Approach #############################
def get_num_tokens(dt_matrix):
    """Input is a document-term matrix of type csr_matrix.sparse. Sums up the number of tokens
    by adding the sum of all rows in the matrix"""

    rows, cols = dt_matrix.get_shape()
    num_tokens = 0
    for i in range(0, rows):
        num_tokens += dt_matrix.getrow(i).sum()

    return num_tokens


def get_topics_w_query(topic_dist, topn, feature_names, query_list):
    """asks the user for a query, returns a list of topic numbers that contain the queried word in their top n
    assoc words, topic_dist is the distribution of words per topic, lda.components_ in main, topn is number of
    words to be considered per topic, features names is the list of indecies mapped to terms"""

    # generate list
    topicid_list = []

    for i in range(0, len(topic_dist)):

        # convert topic_dist from ndarray to list
        list_topic = list(topic_dist[i])

        # map current indicies to freq by changing each element to a tuple of (index, freq)
        for j in range(0, len(list_topic)):
            list_topic[j] = (j, list_topic[j])

        # sort the list of tuples by freq
        list_topic = sorted(list_topic, key=itemgetter(1), reverse=True)

        # slice the list so that it only includes the top n words
        if len(list_topic) > topn:
            list_topic = list_topic[:topn]

        # replace tuples with actual terms
        for j in range(0, len(list_topic)):
            list_topic[j] = feature_names[list_topic[j][0]]

        # if the query term is present in the list of top terms in the topic add it to list
        count = 0
        for j in range(0, len(query_list)):
            if query_list[j] in list_topic:
                count += 1

        if count == len(query_list):
            topicid_list.append(i)

    return topicid_list


def get_segs_w_query(doc_topic_dist, topic_id_list, topn, query_list):
    """This function takes the document topic matrix, the list of relevant topics,
    the top N segments to be returned from each topic, and the list of queried words.
    It returns a list of tuples (topic_id, seg_id_list) which contain the topic id
    and a list of the ids of the N most relevant segments to that topic."""

    seg_list = []
    num_segs = 0

    # go through the relevant topics
    for i in range(0, len(topic_id_list)):

        # get topic dist among documents (a collumn for the doc_topic_dist matrix)
        topic_doc_dist = list(doc_topic_dist[:, topic_id_list[i]])

        # sort topic_doc_dist by the association of doc with topic
        # change topic_doc_dist to tuples
        for j in range(0, len(topic_doc_dist)):
            topic_doc_dist[j] = (j, topic_doc_dist[j])

        # sort the tuples of doc id, association val
        topic_doc_dist = sorted(topic_doc_dist, key=itemgetter(1))

        # get topn n ids of the sorted list of (doc_id, association_val)
        topn = min(topn, len(topic_doc_dist))
        seg_ids = [doc_tuple[0] for doc_tuple in topic_doc_dist[:topn]]
        num_segs += len(seg_ids)

        seg_list.append((topic_id_list[i], seg_ids))

    return seg_list, num_segs


############################# SKLearn-LDA #############################

def print_topics(model, feature_names, n_top_words):
    """Prints the topic information. Takes the sklearn.decomposition.LatentDiricheltAllocation lda model,
    the names of all the features, the number of words to be printined per topic, a list holding the freq
    of each topic in the corpus"""
    print("Top 10 words per topic")
    message_list = []

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % (topic_idx)
        list_feat = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]

        feat_freq = sorted(topic, reverse=True)
        for j in range(0, len(list_feat)):
            message += "%s: %s, " % (list_feat[j], str(round(feat_freq[j], 3)))

        message_list.append(message)
        print(message)
    print()

    return message_list


def get_num_tokens(dt_matrix):
    """Input is a document-term matrix of type csr_matrix.sparse. Sums up the number of tokens
    by adding the sum of all rows in the matrix"""

    rows, cols = dt_matrix.get_shape()
    num_tokens = 0
    for i in range(0, rows):
        num_tokens += dt_matrix.getrow(i).sum()

    return num_tokens


############################# Key Word Search #############################

def key_word_search(text_corpus, query_list):
    """Thanks the full corpus of text and the list of words representing a
    concept query. Returns all the segments containing one of those words.
    Segments are a tuple of """
    segs = []
    seg_ids = []
    for i in range(len(text_corpus)):
        paragraph = text_corpus[i]
        # get all the words in the paragraph
        word_list = [word.lower() for word in word_tokenize(paragraph)]

        # check if the query words are present
        for word in query_list:
            if word in word_list:
                segs.append(paragraph)
                seg_ids.append(i)
                break

    return segs, seg_ids


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


############################# Test Elbow On Topic Doc Dist #############################

def visualize(doc_topic_dist):
    # go through each topic
    for i in range(NUM_TOPICS):
        f = plt.figure(i)
        plt.plot(sorted(doc_topic_dist[:, i], reverse=True))
        cutoff = threshold(sorted(doc_topic_dist[:, i]))
        plt.plot(1, cutoff, 'g', marker='o')
        plt.ylabel("Association")
        plt.xlabel("Document")
        plt.title("Topic %d" % (i))
        f.show()

    input()


if __name__ == "__main__":
    main()
