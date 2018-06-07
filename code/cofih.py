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

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}

    for t in range(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)

        # check for convergence
        if np.array_equal(M, Mnew):
            break

        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
    
def bestkmedoids(D, k, iterations=1000):
    clusters = None
    centroids = None
    mse = None
    for i in range(iterations):
        try:
            m, c = kMedoids(D, k)
        except:
            continue
        
        sqerr = []
        for j in range(k):
            cumsqerr += ((D[m[j]] - D[c[j]])**2).sum()
        newmse = cumsqerr/D.shape[0]
        
        if mse == None:
            centroids = m
            clusters = c
            mse = newmse
        elif mse > newmse:
            centroids = m
            clusters = c
    
    return centroids, clusters


def fK(data, labels, lastalpha=None, lastSK=None, metric = "euclidean"):
    k = len(set(labels))
    Nd = data.shape[1]
    
    if k == 1: return 1, None, None
    
    SK = sum(cdist(data[labels==i].toarray(), data[labels==i].mean(0), metric=metric).sum()**2 for i in range(k))
    
    if k == 2:
        lastSK = 1
        alpha = 1 - 3/(4*Nd)
    elif k > 2:
        alpha = lastalpha + (1-lastalpha)/6
    
    if lastSK == 0: return 1, alpha, None
    if Nd <=1: return None, None, None
    
    return SK/(alpha*lastSK), alpha, SK

def fKTest(data, maxk = 30, metric = "euclidean"):
    alpha = None
    SK = None
    fKs = []
    
    if data.shape[1]<3:
        raise StandardError("Dataset too small (only %d row)." % data.shape[1])

    for k in range(1,min(maxk, data.shape[0]-1)):
        km = KMeans(k, max_iter=50, n_init=5)
        km.fit(data)
        r, alpha, SK = fK(data, km.labels_, SK, alpha, metric=metric)
        yield r

def tscore(mat, labels, referenceclusteridx):
    boolmat = sp.csc_matrix(mat, dtype=bool)
    Ninv = 1/boolmat.sum()
    gvec = np.zeros(boolmat.shape[1], dtype=bool)
    gvec[labels == referenceclusteridx] = True
    
    for col in boolmat.T:
        intersec = (gvec & col).sum()
        yield (intersec - Ninv * gvec.sum() * col.sum() ) / np.sqrt(intersec)

def ppmi(mat, labels, referenceclusteridx):
    boolmat = sp.csc_matrix(mat, dtype=bool)
    N = boolmat.sum()
    gvec = np.zeros(boolmat.shape[1], dtype=bool)
    gvec[labels == referenceclusteridx] = True
    
    for col in boolmat.T:
        pmi = np.log((col&gvec).sum() * N / (gvec.sum() * col.sum()))
        ppmi = pmi if pmi >=0 else 0
        yield ppmi

def tfidf(mat, labels, referenceclusteridx):
    N = len(np.unique(labels))
    cidx = referenceclusteridx
    
    #~ lmat = np.zeros((N, mat.shape[1]), dtype=int)
    #~ for i in range(N):
        #~ lmat[i] = mat[labels==i].sum(0)
        
    lmat = sp.csc_matrix(
        [np.squeeze(np.asarray(mat[labels==i].sum(0))) for i in range(N)],
        dtype=int)
    
    for tf in lmat.T:
        if tf.T[cidx].nnz == 0:
            yield 0
        else:
            idx = np.log(N/tf.nnz)
            yield tf.T[cidx].data[0]*idx

def within_interval(x, mu, invcov, chi22p):
    diff = (x-mu).T
    return np.matmul(np.matmul(diff.T, invcov), diff) <= chi22p

class CoFiH:
    metric = "euclidean"
    assoc_function = "tfidf"
    alpha = 0.1
    
    def __init__(self, matrix):
        """Initialize a CoFiH modeler.
        
        Parameters
        ----------
        matrix:  scipy.sparse.csr.csr_matrix
            Document-term matrix
        """
        self.mat = matrix
        self.set_confidence_bound()
    
    def set_confidence_bound(self, p = 0.78, df = 2):
        self.chi22p = chi2.ppf(p,df=df)
    
    def get_aspects(self, query):
        """Applies CoFiH algorithm to find text documents where the 
        concept expressed in query is present.
        
        query: numpy.ndarray or list or tuple                           Louis: Term being in document as first way to express concept, 
                                                                                get every one that contains word of interst
            Mask or indices of the documents assumed to contain the 
            concept of interest (typically because they contain the 
            word associated with said concept). 
        
        Yields
        ------
        topics: list of sets of integers
            Each set is the list of row indices that represents the 
            extension of a topic.
        """
        
        # Create partial matrix containing only query vectors
        qmat = self.mat[query] #CSR_MATRIX -> every number in query, the respective row is added to qmat
        

        # Remove empty attributes
        qmat = qmat.T[np.squeeze(np.asarray(qmat.sum(0)>0))].T
      	

        
        # Get best k
        k = min(enumerate(fKTest(qmat)), key=itemgetter(1))[0]
        # Get partition
        km = KMeans(k)
        km.fit(qmat)
        
        self.kmeans_labels = km.labels_
        n_docs,n_terms = self.mat.shape
       
        for cidx in range(k):
            if sum(km.labels_==cidx) ==1:
                
                
                bool_list = km.labels_==cidx
                for i in range(0,len(bool_list)):
                    if bool_list[i]:
                       index = i

                cg_index = query[i]
                yield np.where(np.squeeze(cg_index))[-1]
                continue
            elif sum(km.labels_==cidx) == 0 :
                continue
            
            assocfn = globals()[self.assoc_function]
            
            # Get top associated
            topn = int(np.round(self.alpha * n_terms))
            topterms = map(itemgetter(0),
                islice(
                    LazySorted(
                        enumerate(assocfn(self.mat, km.labels_, cidx)),
                        key = itemgetter(1),
                        reverse=True
                    ), topn
                )
            )
            
            # Get reduced space
            cmat = self.mat.T[list(topterms)].T
            
            

            # Get cluster vectors' global indices
            indices = []
            bool_list = km.labels_==cidx
            for i in range(0,len(bool_list)):
                if bool_list[i]:
                    indices.append(i)

            cg_indices = query[indices]
            
            # Get what's needed to construct confidence intervals
            mu = cmat[cg_indices].mean(0) # mean of the cluster
            
            Sigma = np.cov(cmat[cg_indices].todense().T)

            if np.linalg.det(Sigma) == 0.0:
                invcov = Sigma
            else:
            	invcov = np.linalg.inv(Sigma)
            
            # Yield an iterator for all 
            
            yield [ i for i, x in enumerate(cmat) \
                if within_interval(x, mu, invcov, self.chi22p) ]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
