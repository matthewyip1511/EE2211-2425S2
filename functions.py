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

    X = pf.fit_transform(X)

    return X

def rank(X: np.ndarray) -> int:
    return np.linalg.matrix_rank(X)

def inverse(X:np.ndarray) -> np.ndarray:
    return np.linalg.inv(X)

# not needed
'''
def P(X, y, order):
    X = np.array([[1,0,1], [1,-1,1]])
    y = np.array([0, 1])
    ## Generate polynomial features
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    ## dual solution (without ridge)
    w_dual = P.T @ inv(P @ P.T) @ y
    print(w_dual)
    ## primal ridge
    reg_L = 0.0001*np.identity(P.shape[1])
    w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
    print(w_primal_ridge)
'''


