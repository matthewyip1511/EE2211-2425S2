import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def matrix(*rows : list) -> np.ndarray:
    return np.array(rows)

def matrix_with_bias(*rows : list) -> np.ndarray:
    return np.array(list(map(lambda lst: [1] + lst, rows)))

def is_invertible(X : np.ndarray) -> bool:
    return np.linalg.det(X) != 0

def has_left_inverse(X : np.ndarray) -> bool:
    return is_invertible(X.T @ X)

def has_right_inverse(X : np.ndarray) -> bool:
    return is_invertible(X @ X.T)

def get_left_inverse(X : np.ndarray) -> np.ndarray:
    return inv(X.T @ X) @ X.T

def get_right_inverse(X : np.ndarray) -> np.ndarray:
    return X.T @ inv(X @ X.T)
    
if __name__ == "__main__":
    import code
    code.interact(local=globals())




