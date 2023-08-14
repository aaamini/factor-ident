import cvxpy as cp
import numpy as np


def PRESS_statistic(r, X):
    """
    Compute the average PRESS statistic of the regression of r on X.

    Parameters
    ----------
    r : np.ndarray
        Dependent variable matrix. Each column is a different response variable.
    X : np.ndarray
        Independent variable matrix.

    Returns
    -------
    float
        Average PRESS statistic across all response variables.
    """

    # Compute regression coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ r

    # Compute residuals
    residuals = r - X @ beta

    # Compute the hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T

    # Compute the PRESS residuals
    h_diag = np.diag(H)[:, np.newaxis]  # make it a column vector for broadcasting
    PRESS = residuals / (1 - h_diag)

    # Compute the PRESS statistic for each response variable
    PRESS_stat = np.sum(PRESS**2, axis=0)

    # Return the average PRESS statistic
    return np.mean(PRESS_stat)


# # A function that takes r and X and computes the PRESS statistic of the regression of r on X
# def PRESS_statistic(r, X):
#     """
#     Compute the PRESS statistic of the regression of r on X.

#     Parameters
#     ----------
#     r : np.ndarray
#         Dependent variable vector.
#     X : np.ndarray
#         Independent variable matrix.

#     Returns
#     -------
#     float
#         PRESS statistic.
#     """
#     # Regress r on X and find the residuals
#     beta = np.linalg.inv(X.T @ X) @ X.T @ r
#     residuals = r - X @ beta
#     # find the hat matrix
#     H = X @ np.linalg.inv(X.T @ X) @ X.T
#     # find the PRESS residuals
#     PRESS = residuals / (1 - np.diag(H))
#     # find the PRESS statistic
#     PRESS_stat = np.sum(PRESS**2)
#     return PRESS_stat

def top_k_rows_indices(Theta_hat, k):
    """
    Return the indices of the top k rows with the highest l2 norms.
    
    Parameters:
    - Theta_hat: numpy array of shape (m, n)
    - k: int, number of top rows to get
    
    Returns:
    - indices: numpy array of shape (k,), indices of the top k rows
    """
    # Compute the l2 norm for each row
    row_norms = np.linalg.norm(Theta_hat, axis=1)
    
    # Get the indices that would sort the norms
    sorted_indices = np.argsort(row_norms)
    
   # Return the indices of the top k rows
    return sorted_indices[::-1][:k]
# Test
# Theta_hat = np.array([[1, 2, 3],
#                       [4, 5, 6],
#                       [7, 8, 9],
#                       [0, 0, 0]])
# k = 2
# print(top_k_rows_indices(Theta_hat, k))

def smallest_singular_value(A):
    """
    Compute the smallest singular value of the matrix A.

    Parameters:
    - A: numpy array of shape (m, n)

    Returns:
    - float, the smallest singular value of A
    """
    
    # Compute the singular values of A
    singular_values = np.linalg.svd(A, compute_uv=False)
    
    # Return the smallest singular value
    return singular_values[-1]


def create_factors(n, k, p, sig=1, eps=0.1, correlated=False):

    if correlated:
        # Create X as an n x k standard Gaussian matrix
        X = np.random.randn(n, 2)
        
        # Create a matrix whose columns are repeats of the first two columns k times
        X = np.hstack((X, np.repeat(X[:,0:2], k/2 - 1, axis=1))) + eps*np.random.randn(n, k)
    else:
        # Create X as an n x k standard Gaussian matrix
        X = np.random.randn(n, k)

    # Generate B with entries from a normal distribution with mean 0 and variance 1/k
    B = np.random.randn(k, p-k) * np.sqrt(1/k) * np.sqrt(1 - sig**2)

    # Generate W as the noise matrix with entries from a normal distribution with mean 0 and variance sig^2
    W = sig * np.random.randn(n, p-k)

    # Calculate M2
    M2 = X @ B

    # Calculate Y
    Y = M2 + W

    # Calculate Z = [X  Y] (column concatenation of X and Y)
    Z = np.hstack((X, Y))

    # Create the mean matrix
    M = np.hstack((X, M2))

    return Z, M, B

def compute_k_truncated_svd(Z, k):
    """
    Compute the k-truncated SVD of the matrix Z.
    
    Parameters:
    - Z: numpy array of shape (m, n)
    - k: int, number of singular values to consider
    
    Returns:
    - Zk: numpy array, k-truncated approximation of Z
    - svd_C: float, the sum of l2 norms of rows of Theta_tru
    """
    
    # Compute the k-truncated SVD of Z
    U, S, V = np.linalg.svd(Z, full_matrices=False)
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = V[:k, :]
    Zk = Uk @ Sk @ Vk

    # Invert Sk @ Vk
    Theta_tru = np.linalg.inv(Sk) @ Vk[:,:k]
    svd_C = cp.sum(cp.norm(Theta_tru, 2, axis=1)).value
    
    return Zk, Theta_tru, svd_C

def top_k_rows_indices(Theta_hat, k):
    """
    Return the indices of the top k rows with the highest l2 norms.
    
    Parameters:
    - Theta_hat: numpy array of shape (m, n)
    - k: int, number of top rows to get
    
    Returns:
    - indices: numpy array of shape (k,), indices of the top k rows
    """
    # Compute the l2 norm for each row
    row_norms = np.linalg.norm(Theta_hat, axis=1)
    
    # Get the indices that would sort the norms
    sorted_indices = np.argsort(row_norms)
    
   # Return the indices of the top k rows
    return sorted_indices[::-1][:k]


# ZK is the k-truncated SVD of Z
def tsms(Z, Zk, knum):
    ZZk_norms = col_norms(Z - Zk)
    return np.argsort(ZZk_norms)[:knum]

def svd_convex_optimization(Z, svd_C, knum):
    # Define the dimensions of the matrices
    n, p = Z.shape
    # SVD Z and define the knum principal components
    U = np.linalg.svd(Z)[0][:,:knum]
    # Define the optimization variable Gamma
    Gamma = cp.Variable((p, knum))
    # Define the objective function
    objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    # Define the constraint
    constraints = [
        cp.sum(cp.norm(Gamma, 2, axis=1))<=svd_C
    ]
    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    # Solve the optimization problem
    result = problem.solve()
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value


# calculate the column norms of Z
def col_norms(Z):
    """
    Calculate the column norms of Z.

    Parameters
    ----------
    Z : np.ndarray
        Concatenated factor matrix.

    Returns
    -------
    np.ndarray
        Column norms of Z.
    """
    return np.sqrt(np.sum(Z**2, axis=0))

# a function to normalize columns of a matrix
def normalize_columns(A):
    # Calculate the norms of the columns of A
    norms = np.linalg.norm(A, axis=0)
    
    # Normalize the columns of A    
    return A / norms
