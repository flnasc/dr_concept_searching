# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:39:25 2018

@author: James

Code for determining a threshold value that
minimizes intra-class variance (equivalent to
maximizing inter-cass variance). 
 
We consider two classes, high relevance and low relevance, 
based on the pseudocounts returned by the LDA 
assignment of words to topics.

Based on Otsu's method
"""
import numpy as np
import matplotlib.pyplot as plt

evil = [485, 129, 126, 100, 96, 88, 80, 63, 59, 52]
symbol = [227, 185, 100, 87, 73, 73, 70, 68, 57, 54]
test1 = [400, 401, 402, 350, 55, 34]
test2 = [200, 189, 178, 150, 45, 50]

'''
The input for the threshold function is an array of pseudocounts. (This can be changed)
The ouput is a threshold (cutoff value); 
words with pseudocounts less than this threshold are
deemed as low relevance for the topic (concept). 
'''
def threshold(pseudocounts):
	number_of_tokens = len(pseudocounts)
	intra_class_variance = np.ones(number_of_tokens)
	counter = 0  
	for t in pseudocounts:     
		high_relevance_slice = [i  for i in pseudocounts if i >= t]
		low_relevance_slice = [i  for i in pseudocounts if i < t]
		high_relevance_probability = sum(high_relevance_slice)/sum(pseudocounts)
		#high_relevance_probability = len(high_relevance_slice)/number_of_tokens
		low_relevance_probability = 1 - high_relevance_probability
		if len(high_relevance_slice):
			high_relevance_variance = np.var(high_relevance_slice)
		else:
			high_relevance_variance = 0
		if len(low_relevance_slice):
			low_relevance_variance = np.var(low_relevance_slice)
		else:
			low_relevance_variance = 0
		intra_class_variance[counter] = low_relevance_probability*low_relevance_variance+high_relevance_probability*high_relevance_variance
		counter += 1
	#return(pseudocounts[np.argmin(intra_class_variance)], intra_class_variance)
	return(pseudocounts[np.argmin(intra_class_variance)])  

def limit_by_threshold(array, threshold):
	"""takes the given array and returns a sliced version up to the given thershold"""
	array = sorted(array, reverse=True)

	cutoff = len(array) - 1

	for i in range(len(array)):
		if array[i] < threshold:
			cutoff = i
			break

	return array[:cutoff]
			 
		
	
