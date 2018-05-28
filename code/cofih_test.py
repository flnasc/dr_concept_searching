"""
	Author: Dylan Hayton-Ruffner
	Description: Test the CoFiH class provided by Louis Chartrand on Documents from Digital Ricoeur 
	Status: Imports set up and skeleton started. 
	Todo: Build input Document-Term matrix from data files

"""

import nltk

import cofih as c
import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2
from sklearn.cluster import KMeans
from operator import itemgetter
from itertools import islice
from lazysorted import LazySorted
from scipy.spatial.distance import cdist
import brisk


#************************** Skeleton **************************#


def main():
	doc_string = load_document("../data/evil_a_Challenge_to_Philosophy_and_Theology.txt")
	print(doc_string)

	sent_tokenizer = nltk.sent_tokenize
	doc_sentances = sent_tokenizer(text = doc_string)
	print(doc_sentances)

	return 1

def load_document(filepath):
	"""	
		Description:Opens and loads the file specified by filepath as a raw txt string; assumes valid text file format.
		Input: String -> filepath of file from current directory
		Output: Entire contents of text file as a string

	"""

	assert(filepath.endswith(".txt")), "Function: Load Document -> File specificed by filepath is not of type .txt"
	file = open(filepath, 'r')
	file_string = file.read()
	file.close()
	
	return file_string


main()




