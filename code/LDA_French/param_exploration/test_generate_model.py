import unittest
from generate_models import get_param_sets
class TestStringMethods(unittest.TestCase):

    def test_simple(self):
        w_r = (0,1)
        k_r = (0, 1)
        n_r = (0, 1)
        self.assertEqual(len(get_param_sets(w_r, k_r, n_r)), 1)
        self.assertEqual(get_param_sets(w_r, k_r, n_r)[0], (hash((0,0,0)),0,0,0))

    def test_complex(self):
        w_r = (0,2)
        k_r = (0, 2)
        n_r = (0, 2)
        self.assertEqual(len(get_param_sets(w_r, k_r, n_r)), 8)
        self.assertEqual(get_param_sets(w_r, k_r, n_r)[0], (hash((0,0,0)),0,0,0))
        self.assertEqual(get_param_sets(w_r, k_r, n_r)[-1], (hash((1, 1, 1)), 1, 1, 1))

if __name__ == '__main__':
    unittest.main()