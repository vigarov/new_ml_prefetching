import numpy as np
import unittest

# Mean precision at k
def mpatk_loop_version(gt, preds):
    res = 0
    for i in range(gt.shape[0]):
        res += np.isin(gt[i], preds[i]).sum()/gt.shape[1]/gt.shape[0]
    return res

def mpatk_vectorized_version(gt, preds):
    return (gt[..., None] == preds[:, None, :]).any(axis=-1).sum() / (gt.shape[0] * gt.shape[1])

def success_loop_version(gt, preds):
    res = 0
    for i in range(gt.shape[0]):
        res += int(np.isin(gt[i][0], preds[i]).sum().item())
    return res / gt.shape[0]

def success_vectorized_version(gt, preds):
    return (gt[:, 0, None] == preds).any(axis=1).mean()


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
        # Small arrays with different shapes and value ranges
        self.gt1 = np.random.randint(0, 10, size=(2, 3))
        self.preds1 = np.random.randint(0, 10, size=(2, 3))
        
        self.gt2 = np.random.randint(0, 100, size=(5, 10))
        self.preds2 = np.random.randint(0, 100, size=(5, 10))
        
        self.gt3 = np.random.randint(0, 1000, size=(20, 10))
        self.preds3 = np.random.randint(0, 1000, size=(20, 10))
        
        # Additional test cases with various shapes
        self.gt4 = np.random.randint(0, 50, size=(10, 5))
        self.preds4 = np.random.randint(0, 50, size=(10, 5))
        
        self.gt5 = np.random.randint(0, 200, size=(15, 8))
        self.preds5 = np.random.randint(0, 200, size=(15, 8))
        
        self.gt6 = np.random.randint(0, 500, size=(8, 15))
        self.preds6 = np.random.randint(0, 500, size=(8, 15))
        
        # Large arrays
        self.gt7 = np.random.randint(0, 2000, size=(50, 20))
        self.preds7 = np.random.randint(0, 2000, size=(50, 20))
        
        self.gt8 = np.random.randint(0, 5000, size=(30, 30))
        self.preds8 = np.random.randint(0, 5000, size=(30, 30))
        
        # Small number of rows, many columns
        self.gt9 = np.random.randint(0, 300, size=(3, 50))
        self.preds9 = np.random.randint(0, 300, size=(3, 50))
        
        # Many rows, small number of columns
        self.gt10 = np.random.randint(0, 300, size=(50, 3))
        self.preds10 = np.random.randint(0, 300, size=(50, 3))

    def test_mpatk(self):
        # Test all cases
        for gt, preds in [
            (self.gt1, self.preds1), 
            (self.gt2, self.preds2), 
            (self.gt3, self.preds3),
            (self.gt4, self.preds4),
            (self.gt5, self.preds5),
            (self.gt6, self.preds6),
            (self.gt7, self.preds7),
            (self.gt8, self.preds8),
            (self.gt9, self.preds9),
            (self.gt10, self.preds10)
        ]:
            loop_result = mpatk_loop_version(gt, preds)
            vectorized_result = mpatk_vectorized_version(gt, preds)
            self.assertAlmostEqual(loop_result, vectorized_result, 
                                places=7, 
                                msg=f"Failed for shapes {gt.shape}")
            
    def test_success(self):
        # Test all cases
        for gt, preds in [
            (self.gt1, self.preds1), 
            (self.gt2, self.preds2), 
            (self.gt3, self.preds3),
            (self.gt4, self.preds4),
            (self.gt5, self.preds5),
            (self.gt6, self.preds6),
            (self.gt7, self.preds7),
            (self.gt8, self.preds8),
            (self.gt9, self.preds9),
            (self.gt10, self.preds10)
        ]:
            loop_result = success_loop_version(gt, preds)
            vectorized_result = success_vectorized_version(gt, preds)
            self.assertAlmostEqual(loop_result, vectorized_result, 
                                places=7,
                                msg=f"Failed for shapes {gt.shape}")

    def test_edge_cases(self):
        # Test case where gt and preds are identical
        identical = np.random.randint(0, 100, size=(5, 10))
        self.assertEqual(
            mpatk_vectorized_version(identical, identical), 
            1.0
        )
        self.assertEqual(
            success_vectorized_version(identical, identical), 
            1.0
        )
        
        # Test case with all different values
        gt = np.arange(50).reshape(5, 10)
        preds = np.arange(50, 100).reshape(5, 10)
        self.assertEqual(
            mpatk_vectorized_version(gt, preds), 
            0.0
        )
        self.assertEqual(
            success_vectorized_version(gt, preds), 
            0.0
        )


class TestMetricsExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
    def test_single_element_arrays(self):
        # Test with single-element arrays
        gt_single = np.array([[5]])
        preds_single = np.array([[5]])
        
        self.assertEqual(mpatk_vectorized_version(gt_single, preds_single), 1.0)
        self.assertEqual(success_vectorized_version(gt_single, preds_single), 1.0)
        
    def test_repeated_values(self):
        # Test with repeated values in ground truth
        preds_repeated = np.array([[1, 1, 1], [2, 2, 2]])
        gt_unique = np.array([[1, 3, 4], [2, 5, 6]])
        
        expected_mpatk = 1/3  # Only one match per row, divided by 3 columns
        self.assertAlmostEqual(mpatk_vectorized_version(gt_unique, preds_repeated), expected_mpatk)
        self.assertEqual(success_vectorized_version(gt_unique, preds_repeated), 1.0)
        
    def test_partially_correct_predictions(self):
        # Test with partially correct predictions
        gt = np.array([[1, 2, 3], [4, 5, 6]])
        preds = np.array([[1, 7, 8], [9, 5, 10]])
        
        expected_mpatk = (1/3 + 1/3) / 2  # One match in each row
        self.assertAlmostEqual(mpatk_vectorized_version(gt, preds), expected_mpatk)
        self.assertEqual(success_vectorized_version(gt, preds), 0.5)
        
    def test_large_value_range(self):
        # Test with very large numbers
        gt_large = np.array([[1000000, 2000000], [3000000, 4000000]])
        preds_large = np.array([[1000000, 9999999], [9999999, 4000000]])
        
        expected_mpatk = 0.5  # One match in each row, divided by 2 columns
        self.assertAlmostEqual(mpatk_vectorized_version(gt_large, preds_large), expected_mpatk)
        self.assertEqual(success_vectorized_version(gt_large, preds_large), 0.5)
        
    def test_negative_values(self):
        # Test with negative values
        gt_neg = np.array([[-1, -2, -3], [-4, -5, -6]])
        preds_neg = np.array([[-1, -7, -8], [-9, -5, -10]])
        
        expected_mpatk = (1/3 + 1/3) / 2  # One match in each row
        self.assertAlmostEqual(mpatk_vectorized_version(gt_neg, preds_neg), expected_mpatk)
        self.assertEqual(success_vectorized_version(gt_neg, preds_neg), 0.5)
        
    def test_mixed_values(self):
        # Test with mixed positive and negative values
        gt_mixed = np.array([[-1, 0, 1], [2, -2, 0]])
        preds_mixed = np.array([[-1, 3, 4], [5, -2, 6]])
        
        expected_mpatk = (1/3 + 1/3) / 2  # One match in each row
        self.assertAlmostEqual(mpatk_vectorized_version(gt_mixed, preds_mixed), expected_mpatk)
        self.assertEqual(success_vectorized_version(gt_mixed, preds_mixed), 0.5)

if __name__ == '__main__':
    unittest.main()