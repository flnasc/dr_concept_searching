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
MIN_WORD_COUNT = 0
import bs4 as Soup
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import re
# import matplotlib.pyplot as plt
import numpy as np
import cofih as c
import sys



############################# MAIN #############################

def main():

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	filepath = input("Filepath to desired document: ")
	print("DESIRED FILE: " + filepath)
	doc_string = load_document(filepath)
	doc_soup = Soup.BeautifulSoup(doc_string, 'xml') 
	doc_para = doc_soup.find_all('p') #use beautiful soup to find all contents of the paragraph
	
	#get contents of each paragraph tag and add them to the list 'corpus'
	corpus = []
	lens = []
	corpusWCS = []
	for i in range(0, len(doc_para)):
		numbW = re.findall(r'\b\w+', doc_para[i].get_text())
		corpus.append(doc_para[i].get_text())

  	#Create CountVectorizer to get Document-Term matrix 
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)
	
	#train vectorizer on corpus 
	vectorizer.fit_transform(corpus)

	#get matrix for corpus 
	matrix = vectorizer.transform(corpus)
	
	matDense = matrix.todense()
	rows, cols = matrix.get_shape()

	cofih = c.CoFiH(matrix)
	
	while True:

		#generate query mask for cofih (list containing the row number in the document-term matrix that contains the queried term)
		query = input("Query: ")
		print(query)
		querySet = getQuery(query, matrix, vectorizer.get_feature_names()) 
		
		#create the cofih object with the Document-Term matrix and run the given query
		print("_______RUN COFIH________") 

		result = cofih.get_aspects(querySet)
		indices = next(result)

		end = True
		while end:
			try:
				indices = np.append(indices, next(result))
			except:
				end = False
		indices = np.unique(indices)

		print("NUM SEGS= " + str(len(indices)))
		if len(indices)<100:
			print(indices)

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




############################# STATS ############################# 

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


       
