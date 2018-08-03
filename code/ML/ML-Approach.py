""" 
   Author: Dylan Hayton-Ruffner
   Description: Takes a document in xml format and splits it into paragraphs. Returns the set of paragraphs that contain sentances whose subject
   is a query read-in from the user. 

   
   Status: WIP
   ToDo: 
      
   NOTES:
   	   
   	   
"""
TEST_SET_SIZE = 0.2
MIN_WORD_COUNT = 10
MIN_SEG_COUNT = 10
DEC_FUNC_THRESHOLD = 200
MAX_DF = 0.95
MIN_DF = 2
TOP_N_SEGS=10
MIN_DF = 0.01
MAX_DF = 0.95
FILETYPE = 'xml'
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
	print("\n-----LDA CONCEPT DETECITON-----")
	text_corpus, raw_corpus = load_corpus('v')
	query = ["symbol", "symbols", "symbolism", 'symbolisms'] #concept

	#get the training data
	concepts_raw = load_document(CONCEPTS_PATH)
	concepts = parse_concepts(concepts_raw)
	num_segs = len(text_corpus)
	nlp = spacy.load('en_core_web_sm')
	positives = [] #the segments that contain the query in subject form
	pos_ids = [] #holds segement ids (the index of the segment in the text_corpus list)
	num_w_query = 0

	#find positive segments (instances of concept)


	concept_queries = []
	short_concepts = []
	full_set = []

	for i in range(0,len(concepts)):

		start = time.time()
		positives, pos_ids = search_corpus(text_corpus, concepts[i], nlp)
		if len(positives) > MIN_SEG_COUNT:
			concept_queries.append([concepts[i], positives, pos_ids])
			for j in range(0, len(positives)):
				full_set.append(positives[j])
		end = time.time()
		print(end - start)

	


	for i in range(0,len(concept_queries)):

		positives = concept_queries[i][1].copy()
		pos_ids = concept_queries[i][2].copy()
		negatives = []
		neg_ids = []

		for j in range(0, len(concept_queries)):

			posb_neg = concept_queries[j][1].copy()
			posb_neg_ids = concept_queries[j][2].copy()
			for k in range(0, len(posb_neg_ids)):
				if (posb_neg_ids[k] not in pos_ids) and (posb_neg_ids[k] not in neg_ids) :
					negatives.append(posb_neg[k])
					#print(posb_neg[k])
					neg_ids.append(posb_neg_ids[k])

		
		#print("Negatives, before",negatives)
		training, test, train_target, test_target = build_train_test_sets(positives, negatives, TEST_SET_SIZE)

		
		





		vectorizer = CountVectorizer(stop_words = 'english',lowercase= True, max_df=MAX_DF, min_df=MIN_DF, tokenizer=LemmaTokenizer())
		vectorizer.fit(full_set)	
		training_matrix = vectorizer.transform(training).todense()
		corpus_matrix = vectorizer.transform(text_corpus).todense()
		test_matrix =  vectorizer.transform(test)
		sgd_clf = SGDClassifier(random_state=42)
		#print(concept_queries[i][0],training, train_target)
		sgd_clf.fit(training_matrix, train_target)

		results = []
		res_ids = []
		
		scores = sgd_clf.decision_function(corpus_matrix)
		for k in range(0, len(scores)):
			if scores[k] > DEC_FUNC_THRESHOLD:
				results.append(text_corpus[k])

		print(concept_queries[i][0],len(training), len(test), len(results))

		concept_queries[i].append(results)
		

		write_output_file_xlsx(concept_queries[i], "../../data/symbolism-of-evil.xml", len(text_corpus), MIN_WORD_COUNT,DEC_FUNC_THRESHOLD)




	# write_output_file(results)


	# print(len(negatives), "negatives")
	# print(len(positives), "positives")
	# print(len(results), "results")
	# print(len(training), "training len")
	# print(len(test),"testing len")
	# print(recall_score(test_target, train_pred), "recall")
	# print(precision_score(test_target, train_pred), "prescision")
	# print(f1_score(test_target, train_pred), "f1")
	# print(roc_auc_score(test_target, train_pred), "roc_auc_score")

	# target_scores = cross_val_predict(sgd_clf, training_matrix, train_target, cv=3, method="decision_function")
	# precisions, recalls,thresholds = precision_recall_curve(train_target, target_scores)
	# pre_rec_plot(precisions, recalls, thresholds)
	# plt.show()
	

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

#
############################# NLP ############################# 

def contains_query(para,query, nlp):
	"""Returns a bool representing whether the given paragraph (para -> list of sents) contains a sentance with the query (as subject)"""

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

		doc = nlp(para[i])
		for chunk in doc.noun_chunks:
			
			#check if each noun phrase is the subject
			if chunk.root.dep_ == "csubj" or chunk.root.dep_ == "csubjpass" or chunk.root.dep_ == "nsubj" or chunk.root.dep_ == "nsubjpass":
				if str(chunk.root).lower() in query:
					return True

	return False
def search_corpus(text_corpus, query, nlp):
	"""Searchs the given list of text segments for intances in which a segment contains the query in subject form.
	Returns all instances in a list. Returns a second list of the segment ids (their original index in text_corpus)"""
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
	"""Takes two lists of true positive and true negative instances of the concept and combine them into training and test
	sets. Returns these two sets. Also returns target arrays, for each of the sets. These arrays contain True wherever the 
	corresponding indice of the set has a true positiv value."""

	
	#print("negatives after",negatives)
	for i in range(0, len(positives)):
		positives[i] =  (True, positives[i])




	
	train_pos, test_pos = positives[int(len(positives)*test_set_size):], positives[:int(len(positives)*test_set_size)]
	train_neg, test_neg = negatives[int(len(negatives)*test_set_size):], negatives[:int(len(negatives)*test_set_size)]

	train_pos.extend(train_neg)
	test_pos.extend(test_neg)

	np.random.shuffle(train_pos)
	np.random.shuffle(test_pos)

	train_target = train_pos.copy()
	for i in range(0, len(train_pos)):
		
		#print(train_pos[i],type(train_pos[i]) is tuple)
		train_target[i] = type(train_pos[i]) is tuple
		if type(train_pos[i]) is tuple:
			train_pos[i] = train_pos[i][1]

	test_target = test_pos.copy()
	for i in range(0, len(test_pos)):
		test_target[i] = type(test_pos[i]) is tuple
		if type(test_pos[i]) is tuple:
			test_pos[i] =	test_pos[i][1]

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

