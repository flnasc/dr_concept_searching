"""
	Author: Dylan Hayton-Ruffner
	Description: Test the CoFiH class provided by Louis Chartrand on Documents from Digital Ricoeur 
	Status: Imports set up and skeleton started. 
	Todo: Build input Document-Term matrix from data files

"""


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

test_matrix = sp.csr.csr_matrix((90,100), dtype=int)
cofih = c.CoFiH(test_matrix)
query = []
for value in cofih.get_aspects(query):
	print(value)