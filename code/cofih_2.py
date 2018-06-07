#Author: Dylan Hayton-Ruffner
#Description: My implementation of CoFiH

# A lazy implementation of CoFiH
import sys
sys.stdout.flush()
import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2
from sklearn.cluster import KMeans
from operator import itemgetter
from itertools import islice
from lazysorted import LazySorted
from scipy.spatial.distance import cdist
import brisk

class CoFiH:
	metric = "euclidean"
    assoc_function = "tfidf"
    alpha = 0.1
    
	def __init__(self, matrix):
		self.matrix = matrix
        

	def get_aspects(self, query):
		return 0

