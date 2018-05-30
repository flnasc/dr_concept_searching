""" 
   Author: Dylan Hayton-Ruffner
   Description: Parses am XML document, segmenting it into paragraphs
   Status: Inital Skeletion -> WIP
   ToDo: 
       1.) fix NONETYPE ERROR WITH cv.fit_transform
"""

import bs4 as Soup
import nltk
from sklearn.feature_extraction.text import CountVectorizer

def main():
	filepath = raw_input("Filepath to desired document: ")
	print("DESIRED FILE: " + filepath)
	doc_string = load_document(filepath)
	doc_soup = Soup.BeautifulSoup(doc_string, 'xml') 
	doc_para = doc_soup.find_all('p') #use beautiful soup to find all contents fo the para 

	print(doc_para)
	corpus=  ["Knowing yourself is the beginning of all wisdom",
    "What is a friend? A friend is single soul dwelling in two bodies"]
	vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)
	vectorizer.fit_transform(doc_para)
	print(vectorizer.get_feature_names())




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


       
