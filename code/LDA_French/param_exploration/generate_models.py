from multiprocessing import Pool
import csv
"""
Description: This function builds a list of all the possible combinations of the parameters in w, k, and n
within the specified range. Each combination is represented as a tuple of (id, w, k, n). Id is a unique
number identifying that param combination. Generated as a hash of the tuple (w, k, n)
:param: w_range - (int,int): the range of w to explore. w is the minimum length of a segment.
:param: k_range - (int,int): the range of k to explore. k is the number of topics.
:param: n_range - v: the range of n to explore. n is the number of iterations.
:return: list of param sets (tuple)
"""
def get_param_sets(w_range, k_range, n_range):
    param_sets = []
    for w in range(w_range[0], w_range[1]):
        for k in range(k_range[0], k_range[1]):
            for n in range(n_range[0], n_range[1]):
                param_set = (w, k, n)
                param_sets.append((hash(param_set), w, k, n))
    return param_sets


"""
Description: This function test all the sets of parameters in param_set by running the model and collecting results.
:param: param_sets - list: list of tuple of (id, w, k, n)
:return: results - list: list of list of the results, [id, w, k, n, c_v, cm, cm, ll/token, perplexity, convergence], for each model. 
"""
def run_models(param_sets):
    results = []
    for p_set in param_sets:
        results.append(run_model(p_set))
    return results

"""
Description: breaks list into n-sized chunks
:param: l - list: list to be chunked
:param: n - int: size of each chunk
:return: chunked list
"""
def to_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


"""
Description: Explores the parameter space of the given ranges of w, k and n. Uses multiple workers to speed execution.
:param: w_range - (int,int): the range of w to explore. w is the minimum length of a segment.
:param: k_range - (int,int): the range of k to explore. k is the number of topics.
:param: n_range - v: the range of n to explore. n is the number of iterations.
:param: num_workers - int: the number of worker threads to use.
"""
def explore_param_space(w_range, k_range, n_range, num_workers=5):
    param_sets = get_param_sets(w_range, k_range, n_range)
    pool = Pool(num_workers)
    res = pool.map(run_model, param_sets)
    with open('results.csv', 'w+') as csvfile:
        writer = csv.Writer(csvfile)
        for row in res:
            writer.write(res)






"""
Description: This function runs a model
:param: param_set - (float, int, int, int): The parameters for the model, id, w (min words per seg), k (num topics), n (iter).
:return: a list of the models results - [id, w, k, n, c_v, cm, cm, ll/token, perplexity, convergence]
"""
def run_model(param_set):
    return param_set

if __name__ == '__main__':
    explore_param_space((0,1), (0,1), (0,1), 1)
