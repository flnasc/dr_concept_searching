""" 
   Author: Dylan Hayton-Ruffner
   Description: Parses am XML document, segmenting it into paragraphs
   Status: Implemented paragraph extraction and vectorization -> WIP
   ToDo: 
       1.) work on cos and euclidean dist calcs
       2.) Format Maticies as percent frequencies
   NOTES:
   	   1.) can I use unicode strings with CountVect? -> looks like yes
   	   2.) BS4 xml parser only works if TEI is present
   	   3.) IMPORTANT!!! CURRENTLY CALCULATING PARA SIZE BEFORE TEXT PROCESSING
"""

import bs4 as Soup
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re


def main():

	#get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
	filepath = raw_input("Filepath to desired document: ")
	print("DESIRED FILE: " + filepath)
	doc_string = load_document(filepath)
	doc_soup = Soup.BeautifulSoup(doc_string, 'xml') 
	doc_para = doc_soup.find_all('p') #use beautiful soup to find all contents fo the para 
	
	#get contents of each paragraph tag and add them to a list type 'corpus'
	corpus = []
	corpusWCS = []
	for i in range(0, len(doc_para)):
		print(doc_para[i].get_text())
		numbW = re.findall(r'\b\w+', doc_para[i].get_text())
		corpus.append(doc_para[i].get_text())



	print("----CORPUS----")
  	print(corpus)
  

  	#Use CountVectorizer to get sparse matrix with
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)
	
	vectorizer.fit_transform(corpus)
	cmatrix = vectorizer.transform(corpus)
	cmatrix.asfptype()
	cmatDense = cmatrix.todense().astype(float)
	rows, cols = cmatrix.get_shape()
	
	
	print("----CountVectorizer INFO BEFORE SIZE ADJ----")
	print(cmatDense)
	print(vectorizer.get_feature_names())
	print(type(vectorizer.vocabulary))
	
	#adjust cmatrix for the size of each paragraph (words  now constitute precentages)
	for i in range(0, rows):
		for j in range(0, cols):
			numOcc = cmatDense.item(i,j)
			totalWords = cmatrix.getrow(i).sum()
			cmatDense.itemset((i,j), numOcc/totalWords)
	print(cmatDense)







	return 0


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


main()


       
