import time
from multiprocessing import Pool
import sys
def test(size, num_workers=5):
    print('Running models...')
    print(f'Size {size}, Workers {num_workers}')
    pool = Pool(num_workers)
    s = time.time()
    res = pool.map(run_model, [(1,1)]*size)
    e = time.time()
    print(f'Entire job took {e - s} seconds')

def run_model(nums):
    return nums[0] + nums[1]

if __name__ == '__main__':
    test(int(sys.argv[1]), num_workers=int(sys.argv[2]))