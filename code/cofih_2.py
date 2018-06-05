#Author: Dylan Hayton-Ruffner
#Description: My implementation of CoFiH

class CoFiH:
	metric = "euclidean"
    assoc_function = "tfidf"
    alpha = 0.1
    
	def __init__(self, matrix):
		self.matrix = matrix
        

	def get_aspects(self, query):
		return 0

