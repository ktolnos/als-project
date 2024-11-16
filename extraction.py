
import numpy as np
from numba import jit



@jit(nopython=True)
def compute_complexity_array(win: int, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Helper for get_complexity, which uses numba jit for loop speedup
    
    Parameters
    ----------
    win (int): Window size for calculating cross-correlations.
    m1 (np.ndarray): First signal to be analyzed.
    m2 (np.ndarray): Second signal to be analyzed.

    Returns
    -------
    arr_0 (np.ndarray): Cross-correlation array.

    """
    arr_0 = np.zeros((win, win), np.float64)
    m1 = (m1 - m1.mean()) / (m1.std() * len(m1))
    m2 = (m2 - m2.mean()) / (m2.std())  
    for n1 in range(win):
        for n2 in range(win):
            num = 0.0
            for i in range(len(m1)):
                # num += (m1[i-n1]-m1.mean())*(m2[i-n2]-m2.mean())
                # De-meaning happens above, so it is redundant to do 
                # again. 30x speedup, et~9.4 sec vs 0.4 sec in testing
                num += (m1[i-n1])*(m2[i-n2])
            arr_0[n1, n2] = num
    
    return arr_0