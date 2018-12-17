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
TOP_N_WORDS=0
MIN_DF = 0.00
MAX_DF = 1.00
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
from elbow_criteria import threshold
from elbow_criteria import limit_by_threshold
import matplotlib.pyplot as plt

class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in re.findall(REGEX_PATTERN, doc)]



############################# MAIN #############################

def main():
	print("\n-----LDA CONCEPT DETECITON-----")
	text_corpus, text_corpus_ids, raw_corpus, raw_corpus_ids, filepath = load_corpus('v')

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

	visualize(doc_topic_matrix)

	print("Score: " + str(lda.score(dt_matrix)/get_num_tokens(dt_matrix)))

	#print topics, 10 is the number of words in the topic dist to display (e.g. top 10)
	topic_str_list = print_topics(lda, feature_names, 10)
	run_elbow(lda,feature_names)	

	for i in range(0, len(concepts)):
		query_list = concepts[i]
		topicid_list = get_topics_w_query(topic_term_matrix, TOP_N_WORDS, feature_names, query_list)
		seg_list, num_rel_segs = get_segs_w_query(doc_topic_matrix, topicid_list, 10, query_list)

		if len(seg_list) > 0:
			write_output_file_xlsx(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs, text_corpus)

	

	

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
	lines = concepts_raw.split('\n')
	concepts = []
	for line in lines:
		if len(line.split()) > 0:
			concepts.append(line.split())
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
	raw_corpus_ids = []
	cleaned_corpus = []
	cleanded_corpus_ids = []
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)

	for i in range(0, len(doc_para)):
		raw_corpus.append(doc_para[i].get_text())
		raw_corpus_ids = doc_para[i].get("ID")
		#use vectorizer to count number of significant words in each paragraph
		try:
			vectorizer.fit_transform([doc_para[i].get_text()])
			matrix = vectorizer.transform([doc_para[i].get_text()])

			if matrix.sum() > MIN_WORD_COUNT:
				cleaned_corpus.append(doc_para[i].get_text())
				cleaned_corpus_ids = doc_para[i].get("ID")
			else:
				continue
		except ValueError:
			continue

	return cleaned_corpus, cleaned_corpus_ids, raw_corpus, raw_corpus_ids, filepath

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

def get_segs_w_query(doc_topic_dist, topic_id_list, topn, query_list):
	"""This function takes the document topic matrix, the list of relevant topics,
	the top N segments to be returned from each topic, and the list of queried words.
	It returns a list of tuples (topic_id, seg_id_list) which contain the topic id
	and a list of the ids of the N most relevant segments to that topic."""

	seg_list = [] 
	num_segs = 0

	#go through the relevant topics
	for i in range(0, len(topic_id_list)):
		
		#get topic dist among documents (a collumn for the doc_topic_dist matrix)
		topic_doc_dist =  list(doc_topic_dist[:,topic_id_list[i]])

		#sort topic_doc_dist by the association of doc with topic
		#change topic_doc_dist to tuples
		for j in range(0, len(topic_doc_dist)):
			topic_doc_dist[j] = (j, topic_doc_dist[j])

		#sort the tuples of doc id, association val
		topic_doc_dist = sorted(topic_doc_dist, key=itemgetter(1))

		#get topn n ids of the sorted list of (doc_id, association_val)
		topn = min(topn, len(topic_doc_dist))
		seg_ids = [doc_tuple[0] for doc_tuple in topic_doc_dist[:topn]]
		num_segs += len(seg_ids)

		seg_list.append((topic_id_list[i], seg_ids))

	return seg_list, num_segs

############################# Write To File ############################# 
def write_output_file(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs):

	query_name = "_".join(query_list)
	new_f = open("../../results/"+query_name +"-test.txt", 'w+')

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
			new_f.write("Seg: " +str(seg_list[i][1][j][0]) + " AssociationValue: " + str(seg_list[i][1][j][1]) + "\n\n\n")

def write_output_file_xlsx(query_list, topic_str_list, topicid_list, filepath, num_segs, seg_list, num_rel_segs, text_corpus):
	query_name = "_".join(query_list)

	# Start from the first cell. Rows and columns are zero indexed.
	row = 0
	col = 0

	# Create a workbook and add a worksheet.
	workbook = xlsxwriter.Workbook('../../results/'+query_name+'.xlsx')
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
		worksheet.write(row, col, "Segments")
		col +=1
		worksheet.write(row, col, "Score")
		col = 0
		row +=1

		first_data_row = row + 1

		#write segmets
		for j in range(0,len(seg_list[i][1])):
			worksheet.write(row, col, text_corpus[seg_list[i][1][j]], cell_format)
			row +=1

		last_data_row = row 


	
	# Write a total using a formula.
	worksheet.write(row, 0, 'Total')
	worksheet.write(row, 1, '=SUM(B' + str(first_data_row) + ":B" +str(last_data_row)+ ")")
	
	workbook.close()


############################# SKLearn-LDA ############################# 

def print_topics(model, feature_names, n_top_words):
	"""Prints the topic information. Takes the sklearn.decomposition.LatentDiricheltAllocation lda model, 
	the names of all the features, the number of words to be printined per topic, a list holding the freq
	of each topic in the corpus"""
	print("Top 10 words per topic")
	message_list =[]
	
	for topic_idx, topic in enumerate(model.components_):
		
		message = "Topic #%d: " % (topic_idx)
		list_feat = [feature_names[i]
							for i in topic.argsort()[:-n_top_words - 1:-1]]

		feat_freq = sorted(topic, reverse=True)
		for j in range(0, len(list_feat)):
			message += "%s: %s, " % (list_feat[j],str(round(feat_freq[j], 3)))

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
		#get all the words in the paragraph
		word_list = [word.lower() for word in word_tokenize(paragraph)]

		#check if the query words are present
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

		#get the names of the features in sorted order -> argsort() return sorted indicies
		list_feat = [feature_names[i]
							for i in topic.argsort()[::-1]] #[::-1] reverses list
		
		#get the frequencis of the top words (limited by the threshold function)
		feat_freq = sorted(topic, reverse=True)
		cutoff = threshold(sorted(topic, reverse=True))
		limited_freq = limit_by_threshold(feat_freq, cutoff)
		
		for j in range(len(limited_freq)):
			message += "%s: %s, " % (str(list_feat[j]),str(limited_freq[j]))

		message_list.append(message)
		print(message)
	print()
	
	return message_list

############################# Test Elbow On Topic Doc Dist ############################# 

def visualize(doc_topic_dist):
	#go through each topic
	for i in range(NUM_TOPICS):
		f = plt.figure(i)
		plt.plot(sorted(doc_topic_dist[:,i], reverse=True))
		cutoff = threshold(sorted(doc_topic_dist[:,i]))
		plt.plot(1,cutoff,'g', marker='o')
		plt.ylabel("Association")
		plt.xlabel("Document")
		plt.title("Topic %d" % (i))
		f.show()

	input()

if __name__ == "__main__":
	main()





       
