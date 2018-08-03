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
NUM_TOPICS = 10
TOP_N_SEGS=10
TOP_N_WORDS=4
MIN_DF = 0.01
MAX_DF = 0.95
FILETYPE = 'xml'
CONCEPTS_PATH = "../../data/concepts.txt"
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
         return [self.wnl.lemmatize(t) for t in re.findall(REGEX_PATTERN, doc)]



############################# MAIN #############################

def main():
	print("\n-----LDA CONCEPT DETECITON-----")
	text_corpus, raw_corpus, filepath = load_corpus('v')

	# text_corpus_lemma = lemmatize_corpus(text_corpus, 'v')

	concepts_raw = load_document(CONCEPTS_PATH)
	concepts = parse_concepts(concepts_raw)
	num_segs = len(text_corpus)
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
	topic_str_list = print_topics(lda, feature_names, 10, topic_prev)

	
	
	# for i in range(0, len(concepts)):

	# 	query_list = concepts[i]

	# 	topicid_list = get_topics_w_query(topic_term_matrix, TOP_N_WORDS, feature_names, query_list)
	# 	seg_list, num_rel_segs = get_segs_w_query(doc_topic_matrix, topicid_list, TOPIC_PRESSENCE_THRESHOLD, text_corpus, query_list)
	# 	if len(seg_list) > 0:
	# 		write_output_file_xlsx(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs)

	

	

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

############################# Top-N Approach ############################# 

def get_topics_w_query(topic_dist, topn, feature_names, query_list):
	"""asks the user for a query, returns a list of topic numbers that contain the queried word in their top n
	assoc words, topic_dist is the distribution of words per topic, lda.components_ in main, topn is number of
	words to be considered per topic, features names is the list of indecies mapped to terms"""

	#generate list
	topicid_list = []

	#get key words for each topic from user
	for i in range(0, len(topic_dist)):

		#convert topic_dist from ndarray to list
		list_topic = list(topic_dist[i])

		#map current indicies to freq by changing each element to a tuple of (index, freq)
		for j in range(0, len(list_topic)):
			list_topic[j] = (j, list_topic[j])

		#sort the list of tuples by freq
		list_topic = sorted(list_topic, key=itemgetter(1), reverse=True)

		#slice the list so that it only includes the top n words
		if len(list_topic) > topn:
			list_topic = list_topic[:topn]

		#replace tuples with actual terms 
		for j in range(0, len(list_topic)):
			list_topic[j] = feature_names[list_topic[j][0]]

		#if the query term is present in the list of top terms in the topic add it to list
		count = 0
		for j in range(0, len(query_list)):
			if query_list[j] in list_topic:
				count += 1

		if count == len(query_list):
			topicid_list.append(i)
				

		
	return topicid_list	

		



	return 

def get_segs_w_query(doc_topic_dist, topicid_list, threshold, text_corpus, query_list):

	#a 2d list where list i is a list of all the documents with the query from topic i
	seg_list = [] 
	num_segs = 0
	#go through topics with the query
	for i in range(0, len(topicid_list)):

		#get topic dist among documents
		topic_doc_dist =  doc_topic_dist[:,topicid_list[i]]
		init_seg_list = []

		#get row numbers of documents with a probability above the given threshold
		for j in range(0, len(topic_doc_dist)):
			if topic_doc_dist[j] > threshold:
				init_seg_list.append(j)

		final_seg_list = []
		#check if the document has the given query
		for j in range(0, len(init_seg_list)):
			for k in range(0, len(query_list)):

				if query_list[k] in re.findall(REGEX_PATTERN,text_corpus[init_seg_list[j]]):
					final_seg_list.append(text_corpus[init_seg_list[j]])
					num_segs += 1
					break

		seg_list.append((topicid_list[i], final_seg_list))

	return seg_list, num_segs

def get_kw_segs(kw_per_topic, top_segs, text_corpus):
	"""Top segs, list of segs per topic, kw_per_topic, list of kw of each topic,
	return a 2d list where list i are the segments of that topic with at least 3 of the key words"""


	kw_segs = [] 
	for i in range(0, len(top_segs)):
		kw_segs.append([])

	#iterate through topics
	for i in range(0, len(top_segs)): 
		

		#iterate through segs in topic
		for j in range(0, len(top_segs[i])):

			#get list of all words in curr seg (i,j)
			seg = re.findall(REGEX_PATTERN,text_corpus[top_segs[i][j]])

			#count number of occurances of key words of topic i in the seg
			count = 0
			for k in range(0, len(kw_per_topic[i])):
				
				if kw_per_topic[i][k] in seg:
					count += 1

			if count > 2:
				kw_segs[i].append(text_corpus[top_segs[i][j]])

	return kw_segs

def write_output_file(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs):

	query_name = "_".join(query_list)
	new_f = open("../results/"+query_name +".txt", 'w+')

	new_f.write("-------------- LDA Top-4 Results --------------\n")
	new_f.write("Query: " +query_name + "\n")
	new_f.write("Corpus: "+ filepath+ "\n")
	new_f.write("Number of Segs: "+ str(num_segs) + "\n")
	new_f.write("Number of Topics: "+ str(NUM_TOPICS)+ "\n")
	new_f.write("Max DF: " +str(MAX_DF)+ "\n")
	new_f.write("Min DF: " +str(MIN_DF)+ "\n")
	new_f.write("Topic Pressence Threshold: " +str(TOPIC_PRESSENCE_THRESHOLD)+ "\n\n")

	new_f.write("-------------- LDA Topics --------------\n")
	for i in range(0, len(topic_str_list)):
		new_f.write(topic_str_list[i]+ "\n")

	new_f.write("Relevant Topics: " + str(topicid_list) + "\n\n")

	new_f.write("-------------- Segments by Topic --------------\n")
	new_f.write("Number of Relevant Segs: " + str(num_rel_segs) + "\n\n")


	for i in range(0, len(seg_list)):
		new_f.write("Topic: " +str(seg_list[i][0])+ "\n")

		for j in range(0, len(seg_list[i][1])):
			new_f.write("Seg: " +str(seg_list[i][1][j])+ "\n")

def write_output_file_xlsx(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs):
	query_name = "_".join(query_list)

	# Start from the first cell. Rows and columns are zero indexed.
	row = 0
	col = 0

	# Create a workbook and add a worksheet.
	workbook = xlsxwriter.Workbook('../results/'+query_name+'.xlsx')
	worksheet = workbook.add_worksheet()
	

	cell_format = workbook.add_format()
	cell_format.set_text_wrap()
	worksheet.set_column(0,0, 100)

	# write query
	worksheet.write(row,col, "Query: " + query_name)
	row +=1
	worksheet.write(row, col,"Corpus: "+ filepath+ "\n")
	row +=1
	worksheet.write(row, col,"Number of Segs: "+ str(num_segs) + "\n")
	row +=1
	worksheet.write(row, col,"Number of Topics: "+ str(NUM_TOPICS)+ "\n")
	row +=1
	worksheet.write(row, col,"Max DF: " +str(MAX_DF)+ "\n")
	row +=1
	worksheet.write(row, col,"Min DF: " +str(MIN_DF)+ "\n")
	row +=1
	worksheet.write(row, col,"Topic Pressence Threshold: " +str(TOPIC_PRESSENCE_THRESHOLD)+ "\n\n")
	row +=1
	
	#write topics
	for i in range(0, len(topic_str_list)):
		worksheet.write(row, col, topic_str_list[i])
		row +=1
	
	#write segments header
	row += 2
	worksheet.write(row, col,"Segments Returned by Topic")
	row += 1
	worksheet.write(row, col, "Number of Segments Found: " + str(num_rel_segs))
	row += 2

	# Iterate over the data and write it out row by row.
	for i in range(0, len(seg_list)):

		#write topic header
		worksheet.write(row, col, "Topic "+str(seg_list[i][0]))
		row +=1
		worksheet.write(row, col, "Segment")
		col +=1
		worksheet.write(row, col, "Score")
		col = 0
		row +=1

		first_data_row = row + 1

		#write segmets
		for j in range(0,len(seg_list[i][1])):
			worksheet.write(row, col, seg_list[i][1][j], cell_format)
			row +=1

		last_data_row = row 
		print(first_data_row,last_data_row)


	
	# Write a total using a formula.
	worksheet.write(row, 0, 'Total')
	worksheet.write(row, 1, '=SUM(B' + str(first_data_row) + ":B" +str(last_data_row)+ ")")
	
	workbook.close()


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


       
