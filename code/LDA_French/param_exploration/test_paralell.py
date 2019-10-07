import time
from multiprocessing import Pool
import sys
def test(num_workers=5):
    print('Running models...')
    pool = Pool(num_workers)
    s = time.time()
    res = pool.map(run_model, [0]*1000000)
    e = time.time()
    print(f'Entire job took {e - s} seconds')

def run_model(set):
    return 1

if __name__ == '__main__':
    test(int(sys.argv[1]))