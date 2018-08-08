""" 
   Author: Dylan Hayton-Ruffner
   Description: This program utilizes NLP and machine learning to extract concepts from philosophical texts. The user
   inputs a the filepath of an xml file containing a text. The program splits that text into paragraphs and then (via NLP and ML)
   trains a model to search for each of the concepts list in the text document at  CONCEPT_PATH. For each concept with a significant presence 
   in the document, the program writes an Excel file containing the parameterization of the query, the segments, and basic cell programming for 
   segment scoring.

"""
#percentage of the training data used for the test set
TEST_SET_SIZE = 0.2

#the minimum word count of a paragraph, those below are discared
MIN_WORD_COUNT = 10

#the minimum number of segments that must be returned from the NLP query
MIN_SEG_COUNT = 10

#the threshold for the sgd_clf's decision function, increasing reduces the number of segs returned
DEC_FUNC_THRESHOLD = 200

#vectorizer paramerters
MAX_DF = 0.95
MIN_DF = 2


FILETYPE = 'xml'

#path to the conepts file
CONCEPTS_PATH = "../../data/concepts.txt"

import bs4 as Soup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import numpy as np
import spacy 
import sys
from operator import itemgetter 
from nltk import sent_tokenize     
from nltk import word_tokenize        
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import string
import xlsxwriter
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import time

class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in re.findall(u'(?u)\\b\\w\\w\\w+\\b', doc)]

############################# MAIN #############################
def main():
	print("\n-----ML CONCEPT DETECITON-----")

	#load and evaluate the text
	text_corpus, raw_corpus = load_corpus('v')
	num_segs = len(text_corpus)

	concepts_raw = load_document(CONCEPTS_PATH)
	concepts = parse_concepts(concepts_raw)
	
	#get training data
	nlp = spacy.load('en_core_web_sm')

	#the segments that contain the query in subject form
	positives = [] 

	#holds segement ids (the index of the segment in the text_corpus list)
	pos_ids = [] 


	#a set of arrays where each array holds the information (concet, postives, ids) for the coresponding concept
	concept_queries = []
	#the set of all positives, i.e. the entrie vocabulary
	full_set = []

	#find positive segments (instances of concept) for each concept in the provided file, add to concept_queries
	for i in range(0,len(concepts)):

		start = time.time()

		#use nlp to seach the corpus and return all segments with the query as a subject, also returns the segment ids (index)
		positives, pos_ids = search_corpus(text_corpus, concepts[i], nlp)
		if len(positives) > MIN_SEG_COUNT:
			concept_queries.append([concepts[i], positives, pos_ids])
			for j in range(0, len(positives)):
				full_set.append(positives[j])
		end = time.time()
		print(end - start)

	

	#for each concept, traing the model and use it to find segments within the corpus
	for i in range(0,len(concept_queries)):

		positives = concept_queries[i][1].copy()
		pos_ids = concept_queries[i][2].copy()

		negatives = []
		neg_ids = []

		#go through the other concepts and add to the negatives set all segments no present in the positives (locate true negatives)
		for j in range(0, len(concept_queries)):

			posb_neg = concept_queries[j][1].copy()
			posb_neg_ids = concept_queries[j][2].copy()
			for k in range(0, len(posb_neg_ids)):
				if (posb_neg_ids[k] not in pos_ids) and (posb_neg_ids[k] not in neg_ids) :
					negatives.append(posb_neg[k])
					neg_ids.append(posb_neg_ids[k])

	
		training, test, train_target, test_target = build_train_test_sets(positives, negatives, TEST_SET_SIZE)


		vectorizer = CountVectorizer(stop_words = 'english',lowercase= True, max_df=MAX_DF, min_df=MIN_DF, tokenizer=LemmaTokenizer())
		vectorizer.fit(text_corpus)	#NOTE: used to use full_set instead -> does this make a difference 
		training_matrix = vectorizer.transform(training).todense()
		corpus_matrix = vectorizer.transform(text_corpus).todense()
		test_matrix =  vectorizer.transform(test)
		sgd_clf = SGDClassifier(random_state=42)
		sgd_clf.fit(training_matrix, train_target)

		results = []
		res_ids = []
		
		#get scores for entire corpus and, if above threshold, add to corpus
		scores = sgd_clf.decision_function(corpus_matrix)
		for k in range(0, len(scores)):
			if scores[k] > DEC_FUNC_THRESHOLD:
				results.append(text_corpus[k])

		print(concept_queries[i][0], len(training), len(test), len(results))

		#each array now contains the results of the concept_query
		concept_queries[i].append(results)
		write_output_file_xlsx(concept_queries[i], "../../data/symbolism-of-evil.xml", len(text_corpus), MIN_WORD_COUNT,DEC_FUNC_THRESHOLD)



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
def parse_concepts(concepts_raw):
	"""	
		Description:Takes the raw string from the concepts file and parses it into a list of concepts.
		Each line of the file is returned as a list of words.
		Input: String -> filepath of file from current directory
		Output: List where each index is a list of words corresponding to a line of the file.

	"""
	concepts = concepts_raw.split('\n')
	for i in range(0, len(concepts)):
		concepts[i] = concepts[i].split()
	return concepts
def load_corpus(v):
	"""html and xml are supported"""

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	filepath = input("Filepath to corpus: ")

	if (v):
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

############################# Results Format ############################# 
def write_output_file(seg_list):

	
	new_f = open("../../results/ML_results.txt", 'w+')

	new_f.write("-------------- ML Segments --------------\n")

	for i in range(0, len(seg_list)):
		new_f.write("\nSeg:" +str(seg_list[i])+ "\n")
def write_output_file_xlsx(concept_info, filepath, num_segs, min_seg_len, dec_func_threshold):
	

	# Start from the first cell. Rows and columns are zero indexed.
	row = 0
	col = 0

	# Create a workbook and add a worksheet.
	workbook = xlsxwriter.Workbook('../../results/ML/'+str(concept_info[0][0])+'.xlsx')
	worksheet = workbook.add_worksheet()
	

	cell_format = workbook.add_format()
	cell_format.set_text_wrap()
	worksheet.set_column(0,0, 100)

	# write query
	worksheet.write(row,col, "Query: " + str(concept_info[0]))
	row +=1
	worksheet.write(row, col,"Corpus: "+ filepath+ "\n")
	row +=1
	worksheet.write(row, col,"Number of Segs: "+ str(num_segs) + "\n")
	row +=1
	worksheet.write(row, col,"Min Seg Length: "+ str(min_seg_len)+ "\n")
	row +=1
	worksheet.write(row, col,"Decision Function Threshold: " +str(dec_func_threshold)+ "\n\n")
	row +=1
	
	
	#write segments header
	row += 2
	worksheet.write(row, col,"Segments Returned by Query")
	row += 1
	worksheet.write(row, col, "Number of Segments Found: " + str(len(concept_info[3])))
	row += 2

	worksheet.write(row, col, "Segment")
	col +=1
	worksheet.write(row, col, "Score")
	col = 0
	row +=1

	first_data_row = row + 1
	# Iterate over the data and write it out row by row.
	for i in range(0, len(concept_info[3])):


		
		worksheet.write(row, col, concept_info[3][i], cell_format)
		row +=1

	
		


	last_data_row = row 
	# Write a total using a formula.
	worksheet.write(row, 0, 'Total')
	worksheet.write(row, 1, '=SUM(B' + str(first_data_row) + ":B" +str(last_data_row)+ ")")
	
	workbook.close()

############################# NLP ############################# 
def contains_query(para,query, nlp):
	"""	
		Description:Returns a bool representing whether the given paragraph (para -> list of sents) contains a sentance with the query (as subject).
		Input: para (list of strings) -> a list of the sentences in a paragraph from the text
			   query (list) ->  list of words in the concept query
			   nlp -> the nlp processing information ("en_core_web_sm") for spacy
		Output: Bool

	"""

	#go through each sentence and see if sentence contains a word from the query
	for i in range(0, len(para)):
		words = word_tokenize(para[i])
		words = [word.lower() for word in words]
		word_present = False
		for word in query:
			if word in words:
				word_present = True
				break


		if not word_present:
			continue

		#if the query-word is present, check to see if it is the subject
		doc = nlp(para[i])
		for chunk in doc.noun_chunks:
			
			#check if each noun phrase is the subject
			if chunk.root.dep_ == "csubj" or chunk.root.dep_ == "csubjpass" or chunk.root.dep_ == "nsubj" or chunk.root.dep_ == "nsubjpass":
				if str(chunk.root).lower() in query:
					return True

	return False
def search_corpus(text_corpus, query, nlp):
	"""	
		Description:Searchess the given list of text segments (paragraphs) for intances in which a segment contains the query in subject form..
		Input: text_corpus (list of strings) -> a list of the paragraph in the text
			   query (list) ->  list of words in the concept query
			   nlp -> the nlp processing information ("en_core_web_sm") for spacy
		Output: Bool

	"""

	tagged = []
	tagged_ids = []
	for i in range(0, len(text_corpus)):

		para = sent_tokenize(text_corpus[i])
		if contains_query(para,query, nlp):
			tagged.append(text_corpus[i])
			tagged_ids.append(i)

	assert(len(tagged)==len(tagged_ids)), "ERROR: tagged and tagged_ids have different lengths."

	return tagged, tagged_ids

############################# ML ############################# 	
def build_train_test_sets(positives, negatives, test_set_size):
	"""	
		Description:Builds the training and test sets from the set of true negatives and true positives.
		Input: positives (list) -> list of the paragraphs containing the queried concept.
			   negatives (list) -> list of the paragraphs not containing the queried concept.
			   test_set_size (float) -> percentage of the data to be used in test set
		Output: the training and test sets, along with their target lists which contain "True" at each 
		index with a positive instance of the concept

	"""

	#mark positives
	for i in range(0, len(positives)):
		positives[i] =  (True, positives[i])

	#divide data
	train_pos, test_pos = positives[int(len(positives)*test_set_size):], positives[:int(len(positives)*test_set_size)]
	train_neg, test_neg = negatives[int(len(negatives)*test_set_size):], negatives[:int(len(negatives)*test_set_size)]

	#combine pos and neg data
	train_pos.extend(train_neg)
	test_pos.extend(test_neg)

	#randomize order
	np.random.shuffle(train_pos)
	np.random.shuffle(test_pos)

	#write train_target
	train_target = train_pos.copy()
	for i in range(0, len(train_pos)):
		
		train_target[i] = type(train_pos[i]) is tuple
		if type(train_pos[i]) is tuple:
			train_pos[i] = train_pos[i][1]

	#write test_target
	test_target = test_pos.copy()
	for i in range(0, len(test_pos)):
		test_target[i] = type(test_pos[i]) is tuple
		if type(test_pos[i]) is tuple:
			test_pos[i] =	test_pos[i][1]

	#rename lists
	training = train_pos
	test = test_pos

	return training, test, train_target, test_target

############################# MODEL EVAL ############################# 
def pre_rec_plot(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0,1])
    			
	








main()

