""" 
   Author: Dylan Hayton-Ruffner
   Description: Parses am XML document, segmenting it into paragraphs, run Cofih 
   Status: Implemented paragraph extraction and vectorization -> WIP
   ToDo: 
       1.) Generate cofih input and check its validity
       3.) Cofih edits -> matrix singular, cg_index

   NOTES:
   	   2.) BS4 xml parser only works if TEI is present
   	   4.) Lots of in loop optimization is possible (sloppy variable creation/use of data)
   	   
"""
C_SIM_THRESHOLD = 0.2
MIN_WORD_COUNT = 10
NUM_TOPICS = 5
import bs4 as Soup
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import gensim
from operator import itemgetter 



############################# MAIN #############################

def main():
	print("\n-----LDA CONCEPT DETECITON-----")
	text_corpus, raw_corpus = load_corpus()

	#Create CountVectorizer to get Document-Term matrix 
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)
	
	#train vectorizer on corpus 
	vectorizer.fit_transform(text_corpus)

	#get matrix for corpus 
	dt_matrix = vectorizer.transform(text_corpus)

	wCounts = getWC(dt_matrix)
	print("Number of paragraphs loaded out of total: %d/%d" % (len(wCounts),len(raw_corpus)))
	#drawHist(wCounts)

	#get vocab lsit
	vocab = vectorizer.get_feature_names()

	d = getMCWords(dt_matrix, vocab, 10)
	print("Most Common Words: " + str(d))


	#turn vocab lsit into dict 
	id2word = dict(enumerate(vocab))

	#use gensim util to covert matrix into corpus object
	corpus = gensim.matutils.Sparse2Corpus(dt_matrix, documents_columns=False)
	corpus.dictionary = id2word

	#train lda model on corpus
	lda = gensim.models.LdaModel(corpus, id2word=id2word, num_topics = NUM_TOPICS, alpha='auto')

	#get topics from that model
	topics = lda.show_topics(num_topics=NUM_TOPICS)

	print("\n-----Topics-----")
	for i in range(0, len(topics)):
		print(topics[i])

	# print(dt_matrix.todense())
	# tdDist =  get_tdDist(corpus, NUM_TOPICS, lda, topn=10, onlyIds=True)
	# print(tdDist)

	print("\n-----Topic-Document Distribution-----")
	# for i in range(0, len(tdDist)):
	# 	print("------Topic %d------" % (i))
	# 	print("Word Distribution: %s" % (lda.show_topic(i)))
	# 	printSeg(tdDist[i], text_corpus)

	

	



  	
	
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

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	filepath = input("Filepath to corpus: ")
	print("LOADING FILE: " + filepath)
	doc_string = load_document(filepath)
	doc_soup = Soup.BeautifulSoup(doc_string, 'xml') 
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

############################# GENSIM ############################# 

def get_tdDist(corpus, num_topics, lda, topn=0, onlyIds=False):
	tdDist = []
	for i in range(0, num_topics):
		tdDist.append([])

	for i in range(0, len(corpus)):
		topics = lda.get_document_topics(corpus[i])
		for j in range(0,len(topics)):
			tdDist[j].append((i, topics[j][1]))

	for i in range(0, len(tdDist)):
		tdDist[i] = sorted(tdDist[i], key = itemgetter(1), reverse=True)

	if topn:
		for i in range(0, len(tdDist)):
			if len(tdDist[i]) > topn:
				tdDist[i] = tdDist[i][:topn]

	
	if onlyIds:
		tdDistDocId = tdDist
		for i in range(0, len(tdDist)):
			for j in range(0, len(tdDist[i])):
				tdDistDocId[i][j] = tdDist[i][j][0]

		
		tdDist = tdDistDocId

			



	return tdDist

def printSeg(segList, text_seg_corpus):
	for i in range(0, len(segList)):
		print("Segment %d: %s" % (segList[i],text_seg_corpus[segList[i]]))
		print("")





############################# STATS ############################# 

def getWC(dt_matrix):
	rows,cols = dt_matrix.get_shape()
	wCounts = []
	for i in range(0, rows):
		wCounts.append(dt_matrix[i].sum())
	wCounts = sorted(wCounts)
	return wCounts

def getMCWords(dt_matrix, vocab, topn):

	if topn > len(vocab):
		topn = len(vocab)

	rows, cols = dt_matrix.get_shape()
	perWCounts = []
	topCountDict = {}
	for i in range(0,cols):
		perWCounts.append((vocab[i],dt_matrix.getcol(i).sum()))


	perWCounts = sorted(perWCounts, reverse=True, key=itemgetter(1))[:topn]

	for i in range(0, topn):
		topCountDict[perWCounts[i][0]] = perWCounts[i][1]



	

	return topCountDict






def drawHist(wCounts):
	#generate hist of wcounts
	numbins = max(wCounts)-min(wCounts)
	if numbins == 0:
		numbins = max(wCounts)
	histo = np.histogram(wCounts,numbins)

	fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
	axs.hist(wCounts, bins=numbins)
	plt.xticks(np.arange(min(wCounts), max(wCounts)+1, 10.0))

	plt.show()

def calcStats(wCounts):
	total = float(sum(wCounts))
	mean = total/len(wCounts)
	numer = 0.0
	for i in range(0, len(wCounts)):
		numer += pow((wCounts[i] - mean),2)
	numer = numer/len(wCounts)
	return numer**(1/2.0), mean



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



############################# TOPIC-SEG MATCHING #############################

def queryTopic(tVec, segMatrix):
 	"""Returns list of all rows in the segMatrix which "match" the topic t"""

 	rows, cols = segMatrix.get_shape()
 	matches = []
 	for i in range(0, rows):
 		print(str(i) + " " + str(calcDist(tVec, segMatrix.getrow(i), "cosine")))
 		if calcDist(tVec, segMatrix.getrow(i), "cosine") < C_SIM_THRESHOLD:
 			matches.append(i)
 	return matches



############################# COFIH PREPROCESSING #############################

def getQuery(query, cmatrix, vocabList):
 	"""Retreieves set of segments containing the query str, as list of indecies in cmatrix"""
 	#get index (col) of query in matrix via vocabList
 	qI = None
 	if query not in vocabList:
 		print("getQuery() ERROR: QUERY NOT IN VOCAB")
 		quit()
 	else:
 		qI = vocabList.index(query)

 	rows, cols = cmatrix.get_shape()
 	querySet = []
 	for i in range(0, rows):
 		if cmatrix.getrow(i).getcol(qI).sum() > 0 and cmatrix.getrow(i).sum() > MIN_WORD_COUNT:
 			querySet.append(i)
 			
 	return np.asarray(querySet)


def getSegsInMap(segMap, corpus):
	"""Returns a list of the full text segments in the corpus indicated by the segmap"""
	textSegs = []
	for i in range(0, len(segMap)):
		textSegs.append(corpus[segMap[i]])

	return textSegs

def runCoFiH(segMap, cmatrix):
	cofih = c.CoFiH(cmatrix)
	result = cofih.get_aspects(segMap)
	return result

def printOriginalSegs(indices, corpus):
	for i in range(0, len(indices)):
		print("\n\n")
		print("Paragraph " + str(i))
		print(corpus[int(indices[i])])
		











main()


       
