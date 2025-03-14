import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error

def matrix(*rows : list) -> np.ndarray:
    return np.array(rows)

def matrix_with_bias(*rows : list) -> np.ndarray:
    return np.array(list(map(lambda lst: [1] + lst, rows)))

def is_invertible(X : np.ndarray) -> bool:
    return np.linalg.matrix_rank(X) == X.shape[0]

def has_left_inverse(X : np.ndarray) -> bool:
    return is_invertible(X.T @ X)

def has_right_inverse(X : np.ndarray) -> bool:
    return is_invertible(X @ X.T)

def get_left_inverse(X : np.ndarray) -> np.ndarray:
    return inv(X.T @ X) @ X.T

def get_right_inverse(X : np.ndarray) -> np.ndarray:
    return X.T @ inv(X @ X.T)

def one_hot_encode(X: np.ndarray) -> np.ndarray:
    # Reshape y to be a 2D array
    X = X.reshape(-1, 1)

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # sparse=False returns a dense matrix

    # Apply one-hot encoding to y
    X = encoder.fit_transform(X)

    return X

def polynomial(X: np.ndarray, deg: int) -> np.ndarray:
    pf = PolynomialFeatures(deg)

    X = X.reshape(-1, 1)
    X = pf.fit_transform(X)

    return X
    




