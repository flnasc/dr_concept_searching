""" 
   Author: Dylan Hayton-Ruffner
   Description: Parses am XML document, segmenting it into chunks of size C_SIZE, runs sklearn.LatentDirichletAllocation, gets the topics and top segements
   Status: WIP
   
   ToDo: 1.) Resolve chunking problem -> break before or after txt cleaning, if chunked before cleaning you can't ensure that each chunk has 
   the same number of significant words. If chunked after you cant map each chunk to a raw body of text.
      
   NOTES: NONE
   	   
   	   
"""
TOPIC_PRESSENCE_THRESHOLD = 0.5
C_SIM_THRESHOLD = 0.2
MIN_WORD_COUNT = 10
NUM_TOPICS = 15
TOP_N_SEGS=10
C_SIZE = 250
MIN_DF = 0.01
MAX_DF = 0.95
FILETYPE = 'lxml'
import bs4 as Soup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import numpy as np
import sys
from operator import itemgetter 
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import string
from collections import OrderedDict

class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in re.findall(u'(?u)\\b\\w\\w\\w\\w+\\b', doc)]



############################# MAIN #############################

def main():
	print("\n-----LDA CONCEPT DETECITON-----")
	text_corpus = load_corpus_txt()

	print("MAX_DF: " + str(MAX_DF))
	print("MIN_DF: " + str(MIN_DF))
	print("Number of Segs: %d/%d" % (len(text_corpus), len(raw_corpus)))


	#Create CountVectorizer to get Document-Term matrix 
	vectorizer = CountVectorizer(stop_words = 'english',lowercase= True, max_df=MAX_DF, min_df=MIN_DF, tokenizer=LemmaTokenizer())

	#train vectorizer on corpus 
	dt_matrix = vectorizer.fit_transform(text_corpus)
	

	feature_names = vectorizer.get_feature_names()

	print("Number of Features: " + str(len(feature_names)))

	#initialize model
	lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=400,
								learning_method='batch')

	#train the model on the corpus and get a document topic matrix for the corpus
	doc_topic_matrix = lda.fit_transform(dt_matrix)
	topic_term_matrix = lda.components_

	print("Score: " + str(lda.score(dt_matrix)/get_num_tokens(dt_matrix)))

	#get freq of topics in corpus
	topic_prev = get_topic_prevelance(doc_topic_matrix, NUM_TOPICS, len(text_corpus))

	#print topics
	print_topics(lda, feature_names, 10, topic_prev)

	#get top segs assoc with each topic
	top_segs = get_top_segments(NUM_TOPICS, doc_topic_matrix, TOP_N_SEGS)

	#print top segs
	for i in range(0,len(top_segs)):
		print("Topic: %d : " % i )
		for j in range(0, len(top_segs[i])):
			print("---------")
			print("Seg: " + str(text_corpus[top_segs[i][j]]))

	return 0



############################# LOAD DATA ############################# 

def load_document(filepath):
	"""	
		Description:Opens and loads the file specified by filepath as a raw txt string; assumes valid text file format.
		Input: String -> filepath of file from current directory
		Output: Entire contents of text file as a string

	"""

	#assert(filepath.endswith(".txt")), "Function: Load Document -> File specificed by filepath is not of type .txt"
	file = open(filepath, 'r')
	file_string = file.read()
	file.close()
	
	return file_string

def loadTopics(filepath):

	file = open(filepath, 'r') 
	topics = file.readlines()
	return topics

def load_corpus():
	"""html and xml are supported"""

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	filepath = input("Filepath to corpus: ")
	print("LOADING FILE: " + filepath)
	doc_string = load_document(filepath)
	doc_soup = Soup.BeautifulSoup(doc_string, FILETYPE) 
	doc_para = doc_soup.find_all('p') #use beautiful soup to find all contents of the paragraph
	
	#get contents of each paragraph tag and add them to the list 'corpus'
	raw_corpus = []
	cleaned_corpus = []
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)

	for i in range(0, len(doc_para)):
		raw_corpus.append(doc_para[i].get_text())
		#use vectorizer to count number of significant words in each paragraph
		try:
			vectorizer.fit_transform([doc_para[i].get_text()])
			matrix = vectorizer.transform([doc_para[i].get_text()])

			if matrix.sum() > MIN_WORD_COUNT:
				cleaned_corpus.append(doc_para[i].get_text())
			else:
				continue
		except ValueError:
			continue





	return cleaned_corpus, raw_corpus

def load_corpus_txt(c_size):
	#works with a raw text file
	filepath = input("Filepath to corpus: ")
	print("LOADING FILE: " + filepath)
	doc_string = load_document(filepath)
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)
	preproc = vectorizer.build_analyzer()
	proc_text = preproc(doc_string)
	
	count = -1
	corpus_i = 0
	corpus = []
	n = c_size
	final = [proc_text[i * n:(i + 1) * n] for i in range((len(proc_text) + n - 1) // n )]
		
		

	return final

############################# SKLearn-LDA ############################# 

def get_topic_prevelance(doc_topic_matrix, num_topics, total_num_docs):
	"""Input: doc_topic_matrix, a numpy nd array where each row represents a doc, and each collumn is the assocication
	of the doc with a topic. Num_topics and integer holding the number of topics. Total_num_docs is an int holding the 
	number of docs in the corpus.
	Output: a list where index i represents the prevelance of topic i within the corpus."""

	topic_prev = [0] * num_topics
	for i in range(0, num_topics):
		topic_doc = doc_topic_matrix[:,i]
		for j in range(0, len(topic_doc)):
			if topic_doc[j] > TOPIC_PRESSENCE_THRESHOLD:
				topic_prev[i] +=1
		topic_prev[i] = topic_prev[i]/total_num_docs

	return topic_prev

def print_topics(model, feature_names, n_top_words, topic_prev):
	"""Prints the topic information. Takes the sklearn.decomposition.LatentDiricheltAllocation lda model, 
	the names of all the features, the number of words to be printined per topic, a list holding the freq
	of each topic in the corpus"""
	i = 0
	for topic_idx, topic in enumerate(model.components_):
		message = "%f Topic #%d: " % (topic_prev[i],topic_idx)
		i +=1
		list_feat = [feature_names[i]
							for i in topic.argsort()[:-n_top_words - 1:-1]]
		feat_freq = sorted(topic, reverse=True)
		for j in range(0, len(list_feat)):
			list_feat[j] += " " + str(round(feat_freq[j], 3)) + ","

		message += " ".join(list_feat)

		print(message)
	print()

def get_top_segments(num_topics, doc_topic_matrix, topn):
	"""Takes the num_topics (int), the doc_topic_matrix (numpy.ndarray), and an it specifying the
	number of segements per topic to be printed. Returns a list of lists, where list i has the top 
	docs in the matrix for topic i."""

	top_segs = [0] * num_topics
	for i in range(0, num_topics):
		topic_doc = doc_topic_matrix[:,i]
		topic_dict = dict(enumerate(topic_doc))
		topic_dict = OrderedDict(sorted(topic_dict.items(), key = itemgetter(1), reverse = True))
		seg_list = topic_dict.keys()
		top_segs[i] = list(seg_list)[:topn]




	return top_segs

def print_top_segs(top_segs, text_corpus):
	"""Takes the list of top segs per topic, and prints the corresponding segments from the text corpus"""

	for i in range(0, len(top_segs)):
		print("TOPIC %d" % (i))
		for j in range(0, len(top_segs[i])):
			print(top_segs[i][j])

def get_num_tokens(dt_matrix):
	"""Input is a document-term matrix of type csr_matrix.sparse. Sums up the number of tokens
	by adding the sum of all rows in the matrix"""

	rows, cols = dt_matrix.get_shape()
	num_tokens = 0
	for i in range(0, rows):
		num_tokens += dt_matrix.getrow(i).sum()

	return num_tokens



############################# DIST CALCS ############################# 

def calcDist(v1, v2, metric="euclidean"):
	row1, col1 = v1.get_shape()
	row2, col2 = v2.get_shape()
	assert(row1 == 1), "FORMAT ERROR IN calcDist(): v1 argument is not a (1 x n) CSR matrix"
	assert(row2 == 1), "FORMAT ERROR IN calcDist(): v2 argument is not a (1 x n) CSR matrix"
	assert(col2 == col1), "VECTOR SIZE ERROR IN calcDist(): v1 and v2 are not of the same length, i.e must contain diff attributes"

	
	switcher = {"euclidean": euclideanDist, "cosine": cosineDist}
	func = switcher.get(metric, lambda: "Invalid month")
	return func(v1,v2,col1)  

def euclideanDist(v1, v2, cols):
	sumDif = 0.0
	for i in range(0, cols):
		sumDif += pow((v1.getcol(i).sum() - v2.getcol(i).sum()), 2)

	return sumDif ** 0.5

def cosineDist(v1, v2, cols):
	return scipy.spatial.distance.cosine(v1.toarray(),v2.toarray())



main()


       
