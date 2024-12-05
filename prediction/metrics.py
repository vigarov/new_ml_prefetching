import numpy.typing as npt
import numpy as np

def validate_or_get_K(preds: npt.NDArray,K):
    if K is not None:
        assert K == len(preds)
    else:
        K = len(preds)
    return K

def recall_at_K(preds: npt.NDArray, trues: npt.NDArray, K = None):
    K = validate_or_get_K(preds,K)
    return len(np.intersect1d(preds,trues))/K

def success_at_K(preds: npt.NDArray,true):
    return true in preds