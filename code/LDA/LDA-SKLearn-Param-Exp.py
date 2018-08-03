""" 
   Author: Dylan Hayton-Ruffner
   Description: This program gathers data on different combinations of input parameters (min seg size, # topics, max df and min df).
   Each combination is used to generate a model. Each model is scored (log-likelihood per word). The program returns a sorted list
    of the different combinations. 
   Status: Finished
   ToDo: N/A
 
   NOTES: Parameters and corpus are hard-coded
   	   
   	   
"""
TOPIC_PRESSENCE_THRESHOLD = 0.7
C_SIM_THRESHOLD = 0.2
MIN_WORD_COUNT = 10
NUM_TOPICS = 10
TOP_N_SEGS=10
TOP_N_WORDS=4
C_SIZE = 250
MIN_DF = 0.01
MAX_DF = 0.95
FILETYPE = 'xml'
import bs4 as Soup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import numpy as np
import sys
from operator import itemgetter 
from nltk import word_tokenize  
from nltk import pos_tag          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import string
from collections import OrderedDict
import xlsxwriter

class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in re.findall(u'(?u)\\b\\w\\w\\w+\\b', doc)]



############################# MAIN #############################

def main():
	print("\n-----LDA Param Experiment-----")

	#parameters
	min_word_counts = [10,20,30,40]
	num_topics_list = [5,10,20,30,50]
	max_df_list = [0.90,0.95,1.00]
	min_df_list = [0.001,0.005,0.01]

	data = []
	for min_word_count in min_word_counts:
		for num_topics in num_topics_list:
			for max_df in max_df_list:
				for min_df in min_df_list:
					score, num_segs, num_features = get_score("../../data/symbolism-of-evil.xml", min_word_count, num_topics, max_df, min_df)
					combo = ([min_word_count, num_segs, num_features, num_topics, max_df, min_df],score)
					data.append(combo)
					print("finished", combo)

	data = sorted(data, key=itemgetter(1), reverse=True)

	print("------- Sorted Features List -------- ")
	for i in range(0, len(data)):
		curr_data = data[i]
		print("MWC = %d, Num Segs = %d/%d, Num Features = %d, Num Topics = %d, max_df = %f, min_df = %f --> score = %f" % (curr_data[0][0], curr_data[0][1][0], 
			curr_data[0][1][1], curr_data[0][2], curr_data[0][3], curr_data[0][4], curr_data[0][5], curr_data[1]))
		

	
	

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

def load_corpus(min_word_count, filepath=0):

	"""	
		Description: Gets the filepath of the corpus from the user if not provided. Segments the xml file into paragraphs via Soup.
		Creates a cleaned corpus by removing paragraphs with less than the minimum significant word count. Returns 2 lists
		and a string. 

		Input: filepath -> filepath of file from current directory (str)
			   min_word_count -> minimum # of words per paragraph (int)

		Output: Cleaned corpus -> a list of paragraphs that exceed the min word count (raw strings).
			    Raw corpus -> a list of all paragraphs (raw strings). 
			    Filepath -> the filepath string given to the function.

	"""

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	if not filepath:
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

			if matrix.sum() > min_word_count:
				cleaned_corpus.append(doc_para[i].get_text())
			else:
				continue
		except ValueError:
			continue

	return cleaned_corpus, raw_corpus, filepath

def lemmatize_corpus(text_corpus, v):
	"""Takes the list of raw paragraph strings and returns a list of the same strings after lemmatization"""
	if (v):
		print("lemmatizing corpus...")

	lemma_corpus = []
	for i in range(0, len(text_corpus)):

		#tokenize and tag
		para = text_corpus[i]
		para_token = word_tokenize(para)
		para_tagged = pos_tag(para_token)
		para_lemma = []

		#lemmatize
		wnl = WordNetLemmatizer()
		for j in range(0, len(para_tagged)):
			para_tagged[j] = (para_tagged[j][0],get_wordnet_pos(para_tagged[j][1]))
			word_lemma = wnl.lemmatize(para_tagged[j][0], para_tagged[j][1])
			para_lemma.append(word_lemma)

		#return to str format
		para_lemma_str = " ".join(para_lemma)
		lemma_corpus.append(para_lemma_str)

	return lemma_corpus

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

############################# Param Experiment Funcs ############################# 

def get_score(filepath, min_word_count, num_topics, max_df_, min_df_):
	text_corpus, raw_corpus, filepath = load_corpus(min_word_count, filepath)

	num_segs = len(text_corpus)

	#Create CountVectorizer to get Document-Term matrix 
	vectorizer = CountVectorizer(stop_words = 'english',lowercase= True, max_df=max_df_, min_df=min_df_, tokenizer=LemmaTokenizer())

	#train vectorizer on corpus 
	dt_matrix = vectorizer.fit_transform(text_corpus)
	
	feature_names = vectorizer.get_feature_names()

	#initialize model
	lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=5,
								learning_method='batch')


	#train the model on the corpus and get a document topic matrix for the corpus
	doc_topic_matrix = lda.fit_transform(dt_matrix)

	

	feature_names = vectorizer.get_feature_names()
	num_features = len(feature_names)
	score = lda.score(dt_matrix)/get_num_tokens(dt_matrix)

	return score, (num_segs, len(raw_corpus)), num_features   


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
	message_list =[]
	for topic_idx, topic in enumerate(model.components_):
		message = "%f Topic #%d: " % (topic_prev[i],topic_idx)
		i +=1
		list_feat = [feature_names[i]
							for i in topic.argsort()[:-n_top_words - 1:-1]]
		feat_freq = sorted(topic, reverse=True)
		for j in range(0, len(list_feat)):
			list_feat[j] += " " + str(round(feat_freq[j], 3)) + ","

		message += " ".join(list_feat)
		message_list.append(message)
		print(message)
	print()
	return message_list

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

def get_top_segs_threshold(num_topics, doc_topic_matrix, threshold):
	"""Takes the num_topics (int), the doc_topic_matrix (numpy.ndarray), and an it specifying the
	number of segements per topic to be printed. Returns a list of lists, where list i has the top 
	docs in the matrix for topic i."""

	top_segs = []
	for i in range(0, num_topics):
		top_segs.append([])
	for i in range(0, num_topics):
		topic_doc = doc_topic_matrix[:,i]
		for j in range(0, len(topic_doc)):		
			if topic_doc[j] > threshold:
				top_segs[i].append(j)
		
		
		




	return top_segs

def print_top_segs(top_segs, text_corpus):
	"""Takes the list of top segs per topic, and prints the corresponding segments from the text corpus"""

	for i in range(0, len(top_segs)):
		print("TOPIC %d" % (i))
		for j in range(0, len(top_segs[i])):
			print(text_corpus[top_segs[i][j]])

def get_num_tokens(dt_matrix):	
	"""Input is a document-term matrix of type csr_matrix.sparse. Sums up the number of tokens
	by adding the sum of all rows in the matrix"""

	rows, cols = dt_matrix.get_shape()
	num_tokens = 0
	for i in range(0, rows):
		num_tokens += dt_matrix.getrow(i).sum()

	return num_tokens




main()


       
