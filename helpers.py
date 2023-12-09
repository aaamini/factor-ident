'''
Author: Naixin && naixinguo2-c@my.cityu.edu.hk
Date: 2023-08-15 14:09:12
LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
LastEditTime: 2023-12-09 22:40:27
FilePath: /trylab/factor-ident/helpers.py
Description:
 
'''
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from joblib import Parallel,delayed
from celer import GroupLasso
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from joblib import Parallel,delayed
from scipy.stats import norm
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from scipy.stats import f
import heapq
import itertools
import pandas as pd
from sklearn.preprocessing import normalize
import warnings
import cvxpy as cp
from matplotlib.ticker import AutoMinorLocator
warnings.filterwarnings('ignore')
def judge_star(pvalue1):
    pvalue = 1 - pvalue1
    if pvalue > 0.99:
        output = "$^{***}$"
    elif pvalue > 0.95:
        output = "$^{**}$"
    elif pvalue > 0.90:
        output = "$^{*}$"
    else:
        output = ""
    return output
### non nested model
def BKRS_test(FA, FB):
    
    T=FA.shape[0]
    KA = FA.shape[1]
    muA = np.mean(FA, axis=0)
    if KA ==1:
        WA = (((T - KA - 2)/T)/np.cov(FA.T))[np.newaxis,np.newaxis]
    else:   
        WA = ((T - KA - 2)/T)*np.linalg.pinv(np.cov(FA.T))  # (T - K - 2)/T 调整一下
    theta2A = muA.T@WA@muA  - KA/T

    KB = FB.shape[1]
    muB = np.mean(FB, axis=0)
    WB = ((T - KB - 2)/T)*np.linalg.pinv(np.cov(FB.T)) # (T - K - 2)/T 调整一下
    theta2B = muB.T@WB@muB  - KB/T

    dtheta2 = theta2A-theta2B

    FAd = FA - muA
    FBd = FB - muB
    uA = FAd@WA@muA  # T * 1
    uB = FBd@WB@muB
    dt = 2*(uA-uB)-(uA**2-uB**2)

    vd = dt.T@dt
    vd = vd/T
    tvalue = np.abs(dtheta2)/np.sqrt(vd/T)
    pvalue = 2* (1- norm.cdf(np.abs(tvalue)))
    
    return [pvalue,tvalue]
  

def PY_test(F1, F2, test_asset):

    if test_asset[0][0]:
        X2 = np.hstack((F2, test_asset))
    else:
        X2 = F2

    K = F1.shape[1]
    N = X2.shape[1]
    T = X2.shape[0]
    ones = np.ones((T, 1))
    M_F1 = np.eye(T) - F1 @ np.linalg.pinv(F1.T @ F1) @ F1.T
    hat_alpha2 = X2.T @ M_F1 @ ones @ np.linalg.pinv(ones.T @ M_F1 @ ones)

    full_X = np.hstack((ones, F1))
    M_F = np.round(np.eye(T) - full_X @ np.linalg.pinv(full_X.T @ full_X) @ full_X.T,25)


    error = X2.T @ M_F
    error_Sigma = error @ error.T / (T - K - 1)
    m = np.linalg.pinv(ones.T @ M_F1 @ ones)
    var_alpha2 = np.diag(m) * error_Sigma
    T_stat2 = hat_alpha2**2 /(np.diag(var_alpha2)[:,np.newaxis])

    deg = T - K - 1

    rho2 = 0
    Rut = np.corrcoef(error)

    pn = 0.1 / (N - 1)
    thetan = (norm.ppf(1 - pn / 2)) ** 2
    rho2 = (np.sum((Rut[Rut ** 2 * deg > thetan]) ** 2) - N) / 2
    rho2 = rho2 * 2 / N / (N - 1)

    PYk = N ** (-1 / 2) * (np.sum(T_stat2 - deg / (deg - 2))) / ((deg / (deg - 2)) * np.sqrt(2 * (deg - 1) / (deg - 4) * (1 + N * rho2)))

    pvalue = 2 * (1 - norm.cdf(np.abs(PYk)))

    return [pvalue, PYk]


def GRS_test(F1, F2, test_asset, Lambda):
  
    if test_asset[0][0]:
        Y = np.concatenate((F2, test_asset), axis=1)
    else:
        Y = F2
    K = F1.shape[1]
    N1 = Y.shape[1]
    T = Y.shape[0]
    
    # GRS statistic
    if F1.shape[1]==1:
        W1 = 1/np.cov(F1.T)
        mu1 = np.mean(F1)
        SR1 = 1 + mu1.T * W1 * mu1
    else:    
        W1 = np.linalg.pinv(np.cov(F1.T))
        mu1 = np.mean(F1, axis=0)
        SR1 = 1 + mu1.T @ W1 @ mu1
    
    if (N1 + K) < T:
        F = np.concatenate((Y, F1), axis=1)  
    else:
        F = np.concatenate((F2, F1), axis=1)  
    W = np.linalg.pinv(np.cov(F.T))
    mu = np.mean(F, axis=0)
    SR = 1 + mu.T @ W @ mu
    N = F.shape[1] - K
    
    GRSi = ((T - N - K) / N) * (SR / SR1 - 1)
    qf0i = f.ppf(1 - Lambda, N, T - N - K)
    pvalue = 1 - f.cdf(GRSi, N, T - N - K)
    
    return [pvalue, GRSi, qf0i]

# def PRESS_statistic(r, X):
#     """
#     Compute the average PRESS statistic of the regression of r on X.

#     Parameters
#     ----------
#     r : np.ndarray
#         Dependent variable matrix. Each column is a different response variable.
#     X : np.ndarray
#         Independent variable matrix.

#     Returns
#     -------
#     float
#         Average PRESS statistic across all response variables.
#     """
#     # Compute regression coefficients
#     beta = np.linalg.inv(X.T @ X) @ X.T @ r

#     # Compute residuals
#     residuals = r - X @ beta

#     # Compute the hat matrix
#     H = X @ np.linalg.inv(X.T @ X) @ X.T

#     # Compute the PRESS residuals
#     h_diag = np.diag(H)[:, np.newaxis]  # make it a column vector for broadcasting
#     PRESS = residuals / (1 - h_diag)

#     # Compute the PRESS statistic for each response variable
#     PRESS_stat = np.sum(PRESS**2, axis=0)

#     # Return the average PRESS statistic
#     return np.mean(PRESS_stat)
def PRESS_statistic(r, X):

    # Compute regression coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ r

    # Compute residuals
    residuals =( r - X @ beta)[:, np.newaxis]

    # Compute the hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T

    # Compute the PRESS residuals
    h_diag = np.diag(H)[:, np.newaxis]  # make it a column vector for broadcasting
    PRESS = residuals / (1 - h_diag)
    # R= np.sum((r-np.mean([np.setdiff1d(r,i) for i in range(r.shape[0])]))**2, axis=0)
  
    # Compute the PRESS statistic for each response variable
    PRESS_stat = np.sum(PRESS**2)

    # Return the average PRESS statistic
    return PRESS_stat

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
def compute_svd_C(Z, k):
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
   
    Zk =Z[:, :k]

    # Invert Sk @ Vk
    Theta_tru = np.linalg.inv(Zk.T@Zk)@Zk.T @ Z
    svd_C = cp.sum(cp.norm(Theta_tru, 2, axis=1)).value
    
    return svd_C

def create_Z(n, k, p, rho, sig):
    # Create X as an n x k standard Gaussian matrix
    X = np.random.randn(n, k)
    
    #Normalize X
    X = X / np.linalg.norm(X,axis=0)
    
    
    # Create B as a random k x (p-k) matrix
    B = np.random.randn(k, p-k)
    
    # Normalize B such that its l1 matrix norm is rho
    B = normalize(B, axis=0, norm='l2') * rho
    
    # Create W as an n x (p-k) Gaussian matrix
    W = np.random.randn(n, p-k)
    
    # Calculate X B 
    XB = X @ B 

    # Calculate Y = X B + sig*W with normalized XB
    # Y = XB/ np.linalg.norm(XB ,axis=0)  + sig*W
    Y = XB + sig*W
    # Calculate Z = [X  Y] (column concatenation of X and Y)
    Z = np.hstack((X, Y))

    
    Z = Z / np.linalg.norm(Z,axis=0)
    
    
    return Z


def compute_k_truncated_Z(Z, k):
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
  
    Zk = Z[:, :k]

    # Invert Sk @ Vk
    Theta_tru = np.linalg.inv(Zk.T@Zk) @ Zk.T@Z
    svd_C = cp.sum(cp.norm(Theta_tru, 2, axis=1)).value
    
    return Zk, Theta_tru, svd_C


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
def sr(wt, mu, Sigma):
    return (np.dot(wt, mu) ** 2) / np.dot(np.dot(wt, Sigma), wt)     
def sr_best(X,lambda_val = 0.033, sigma = 500, T = 5000):  
    init = 10
    
    # df = pd.read_csv('empirical_data/Fama285_22.csv')
    # Y = df[df.columns[1:]]
    #calculate sample mean and sample covariance matrix
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X.T)
    #solve the optimal MVE weight for all factors
    wall = np.linalg.solve(Sigma, mu)
    wall = wall / np.linalg.norm(wall)
    idx = np.argsort(wall)[::-1]
    #sort factors, from important for SR to less important
    Ssort = idx[:X.shape[1]]
    #select first 20 as initialization
    St = idx[:init]

    
    #number of MH updates
   
    #record the value of sharpe ratio + penalty in each MH update
    SRwPenalty = np.zeros(T)
    srsrsr = np.zeros(T)
    #record the subset selection in each MH update
    Sseq = np.zeros((X.shape[1], T))
    #An very important hyperparameter. Roughly speaking, if the variable can improve the SR by this value, then I can be selected.
    # lambda_val = 0.06
    
    #If sigma is large, then the MH updates are more stable. 
    

    for t in range(T):
        #randomly select a variable to update
        i = np.random.randint(X.shape[1])
        #update the new subset of selected variables
        Snew = np.setdiff1d(np.union1d(i, St), np.intersect1d(i, St))
        #find weight for MVE sharpe ratio w.r.t. St
    
        wt = np.linalg.solve(Sigma[St][:, St], mu[St])
        #find weight for MVE sharpe ratio w.r.t. Snew
        wnew = np.linalg.solve(Sigma[Snew][:, Snew], mu[Snew])
        #Find the "likelihood" ratio. If Snew gives larger SR than St or Snew has less factors than St, then this value will be large. 
        llhratio = np.exp(sigma * (sr(wnew, mu[Snew], Sigma[Snew][:, Snew]) - sr(wt, mu[St], Sigma[St][:, St]) - lambda_val * (len(Snew) - len(St))))
        #MH update. 
        random_num = np.random.rand()
        if llhratio > random_num:
            St = Snew
        #record the selected factors in Sseq
        Sseq[Snew, t] = 1
        #record Sharpe ratio + penalty
        SRwPenalty[t] = sr(wnew, mu[Snew], Sigma[Snew][:, Snew]) - lambda_val * len(Snew)
        srsrsr[t] = sr(wnew, mu[Snew], Sigma[Snew][:, Snew])
    print('sr_best',sr(wt, mu[St], Sigma[St][:, St]))
    
    return St
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
    #####################################################################################
    # objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    # Define the constraint
    constraints = [
        cp.sum(cp.norm(Gamma, 2, axis=1))<=svd_C
    ]
    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    # Solve the optimization problem
    result = problem.solve(solver = cp.MOSEK)
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value
def Z_convex_optimization(Z, svd_C, knum):
    # Define the dimensions of the matrices
    n, p = Z.shape

    # Define the optimization variable Gamma
    # Gamma = cp.Variable((p, p))
    # Z0 = Z[:,St]
    # Gamma0 =np.random.normal(0.001,0.01,(p,p))
    # Gamma0[St] = np.linalg.pinv(Z0.T@Z0)@Z0.T@Z
    
    Gamma = cp.Variable((p,p))
    # Gamma.value = Gamma0
    # Define the objective function
    #####################################################################################
    # objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    objective = cp.Minimize(cp.norm(Z - Z @ Gamma, 'fro')**2)
    # Define the constraint
    constraints = [
        cp.sum(cp.norm(Gamma, 2, axis=1))<=svd_C
    ]
    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    # Solve the optimization problem
    result = problem.solve(solver = cp.MOSEK)
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value

def  adjSGL_convex_optimization(Z, svd_C, knum):
    # Define the dimensions of the matrices
    n, p = Z.shape
    alpha_hat = np.zeros(p)
    Z_hat = np.zeros_like(Z)
    for i in range(p):      
        # Z_minus_i = np.delete(Z * (np.abs(np.mean(Z,axis = 0))/np.std(Z,axis = 0)), i, axis=1)
        Z_minus_i = np.delete(Z, i, axis=1) 
        Ui,Si,Vi = np.linalg.svd(Z_minus_i)
        X = np.hstack((np.ones((n, 1)), Ui[:,:knum]))
        # X = np.hstack((np.ones((n, 1)),Z_minus_i ))
        beta = np.linalg.lstsq(X, Z[:, i], rcond=None)[0]
        alpha_hat[i]=beta[0]
        Z_hat[:, i] = np.dot(X, beta)
    
    Sigma = np.cov(Z - Z_hat, rowvar=False)
    S,U=np.linalg.eig(Sigma)
    #S = np.abs(S)
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_half = U@np.diag(np.sqrt(S))@U.T
   
    alpha_z_score = np.abs(alpha_hat) / np.diagonal(Sigma_half)
    #Gamma = cp.Variable((p+1,p))

    # B = cp.Variable((p+1,p))
    B = cp.Variable((p,knum))
    # B.value =  Gamma0 @ Sigma_half
    Z1 = np.concatenate((np.ones((n,1)),Z ),axis=1)
    ######################### OPT #####################################
    U,S,V=np.linalg.svd(Z)
    Opt_matrix1 = U[:,:knum]
    Opt_matrix2 = Z  
    objective = cp.Minimize(cp.norm(Opt_matrix1 - Opt_matrix2@ B, 'fro')**2 )
    constraints = [
        cp.sum(cp.norm(np.diag(1/alpha_z_score**2)@(B) , 2, axis=1))<=svd_C
    ]
    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    # Solve the optimization problem
    result = problem.solve(solver = cp.MOSEK)
    return result, B.value

def  adjDGL_convex_optimization(Z, svd_C, knum):
    # Define the dimensions of the matrices
    n, p = Z.shape
    alpha_hat = np.zeros(p)
    Z_hat = np.zeros_like(Z)
    for i in range(p):      
      
        Z_minus_i = np.delete(Z, i, axis=1) 
        Ui,Si,Vi = np.linalg.svd(Z_minus_i)
        X = np.hstack((np.ones((n, 1)), Ui[:,:knum]))
        # X = np.hstack((np.ones((n, 1)),Z_minus_i ))
        beta = np.linalg.lstsq(X, Z[:, i], rcond=None)[0]
        alpha_hat[i]=beta[0]
        Z_hat[:, i] = np.dot(X, beta)
    
    Sigma = np.cov(Z - Z_hat, rowvar=False)
    S,U=np.linalg.eig(Sigma)

    Sigma_half = U@np.diag(np.sqrt(S))@U.T
   
    alpha_z_score = np.abs(alpha_hat) / np.diagonal(Sigma_half)
  
    B = cp.Variable((p,p))
  
    ######################### OPT #####################################
    Opt_matrix1 = Z
    Opt_matrix2 = Z  
    objective = cp.Minimize(cp.norm(Opt_matrix1 - Opt_matrix2@ B, 'fro')**2 )
    constraints = [
        cp.sum(cp.norm(np.diag(1/alpha_z_score**2)@(B) , 2, axis=1))<=svd_C
    ]
    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    # Solve the optimization problem
    result = problem.solve(solver = cp.MOSEK)
    return result, B.value

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

def standardize_columns(A):
    # Calculate the mean of the columns of A
    means = np.mean(A, axis=0)
    #calculate the standard deviation of the columns of A
    sigmas = np.std(A, axis=0)
    
    # Normalize the columns of A    
    return (A-means) / sigmas



def train_test(Z,size):
    # n, p = Z.shape
    # shuffle_list = list(range(n))
    # random.shuffle(shuffle_list)
    # Z = np.array([Z[i] for i in shuffle_list])
   
    return Z[:size],Z[size:]

def get_U(Z,knum):
    U = np.linalg.svd(Z)[0][:,:knum]
    return U
# def chosen_set_with_press(Gammavalue,nonzero,Znorm):
#     n,p= Znorm.shape
#     chosen_set = list(np.argsort(-np.round(cp.norm(Gammavalue, 2, axis=1).value,6))[:nonzero])
#     hand_ssr = np.mean([PRESS_statistic_cv(Znorm[:,i],Znorm[:,chosen_set]) for i in np.setdiff1d(range(p),chosen_set)])
#     # hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,chosen_set]) for i in np.setdiff1d(range(p),chosen_set)])
#     return chosen_set,hand_ssr

def chosen_set_with_press(Gammavalue,nonzero,Znorm):
    n,p= Znorm.shape
    chosen_set = list(np.argsort(-np.round(cp.norm(Gammavalue, 2, axis=1).value,6))[:nonzero])
    hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,chosen_set]) for i in np.setdiff1d(range(p),chosen_set)])
    
    # hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,chosen_set]) for i in np.setdiff1d(range(p),chosen_set)])
    return chosen_set,hand_ssr
def chosen_set_with_sr_press(Gammavalue,nonzero,Znorm):
    n,p= Znorm.shape
    
    alpha_hat = np.zeros(p)
    Z_hat = np.zeros_like(Znorm)
    chosen_set = list(np.argsort(-np.round(cp.norm(Gammavalue, 2, axis=1).value,6))[:nonzero])
    for i in range(p):      
      
        Z_minus_i = np.delete(Znorm, i, axis=1) 
        Ui,Si,Vi = np.linalg.svd(Z_minus_i)
        X = np.hstack((np.ones((n, 1)), Ui[:,:nonzero]))
        # X = np.hstack((np.ones((n, 1)),Z_minus_i ))
        beta = np.linalg.lstsq(X, Znorm[:, i], rcond=None)[0]
        alpha_hat[i]=beta[0]
        Z_hat[:, i] = np.dot(X, beta)
    
    Sigma = np.cov(Znorm - Z_hat, rowvar=False)
    S,U=np.linalg.eig(Sigma)

    Sigma_half = U@np.diag(np.sqrt(S))@U.T
   
    alpha_z_score = np.abs(alpha_hat) / np.diagonal(Sigma_half)
   
    hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,chosen_set]@ np.diag(alpha_z_score[chosen_set])) for i in np.setdiff1d(range(p),chosen_set)])
    # hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,chosen_set]) for i in np.setdiff1d(range(p),chosen_set)])
    return chosen_set,hand_ssr
def choose_factor(method, Z, Knum, train_size=600, svd_C = None, asset= [[None]], fix_true = None,fix_false = None,hyper_p = None):
   
    Z = Z[:train_size]
    Znorm = normalize_columns(Z)
    n, p = Z.shape
    # SVD Z and define the knum principal components
    Handssr = []
    if method == 'SGL':
        # shuffle_list = list(range(p))
        # random.shuffle(shuffle_list)
        # Znorm = np.array([Znorm[:,i] for i in shuffle_list]).T
        chosen_set =[]
        Znorm_train,Znorm_test = train_test(Znorm,train_size*3//5)
        # knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        knum_list = np.array([Knum])
        svd_C_list = np.linspace(0, svd_C, 30)
        # svd_C_list = [svd_C]
        # knum_list = np.array([Knum])
        
        SSR = 999
        final_knum = 0
        final_chosen_set = []
        knum_chosen_set = []
        cv_ssr = SSR-1
        final_svd_C  = 0
        
      
        for knum in knum_list:
            for svd_C_i in svd_C_list:
                if asset[0][0]:
                    asset = asset[:train_size]
                    asset = normalize_columns(asset)
                    asset_train,asset_test = train_test(asset,train_size*2//3)
                    U = get_U(np.hstack((Znorm,asset)), knum)
                    # U_train = get_U(np.hstack((Znorm_train,asset_train)), knum)
                    # U_test = get_U(np.hstack((Znorm_test,asset_test)), knum)
                else:
                    U = get_U(Znorm, knum)
                    # U_train = get_U(Znorm_train,knum)
                    # U_test = get_U(Znorm_test,knum)
                
                Gammavalue = svd_convex_optimization(Znorm, svd_C_i, knum)[1]
                
                # Gammavalue = svd_convex_optimization(Znorm_train, svd_C_i, U_train, knum)
                ###########
                
                if fix_true:
                    Gammavalue[fix_true] = np.ones((knum))*100 
                if fix_false:
                    Gammavalue[fix_false] = np.zeros((knum)) 
                
                nonzero = np.count_nonzero(np.round(cp.norm(Gammavalue, 2, axis=1).value,4))
                if nonzero == 0:
                    pass
                elif  0<nonzero<knum :
                    # chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm)
                elif nonzero >= knum:
                    # reg = LinearRegression().fit(Znorm_train[:,[0,1,2,3,4]], Znorm_train[:,np.setdiff1d(range(p),[0,1,2,3,4])])
                    # Znorm_pred = reg.predict(Znorm_test[:,[0,1,2,3,4]])
                    # hand_ssr =sum(sum((Znorm_test[:,np.setdiff1d(range(p),[0,1,2,3,4])]-Znorm_pred)**2))/(p-5)
                    # print('[0,1,2,3,4]:', hand_ssr)
                    
                    
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
                    # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm)
                    
                    # chosen_set,hand_ssr=chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
                  
               
                # print('chosen_set',chosen_set,'cv_ssr',cv_ssr)
                
                    # print('knum',knum,chosen_set)
                if cv_ssr<SSR:
                    SSR = cv_ssr
                    final_chosen_set =  chosen_set
                   
                    final_knum = knum
                    final_svd_C = svd_C_i
            
                    # print('knum',final_knum,'by hand', SSR ,svd_C_i)
    # return SSR,final_chosen_set,final_knum,final_svd_C
            
            if  len(knum_chosen_set)==0:
                knum_chosen_set = final_chosen_set
            elif len(final_chosen_set) >= len(knum_chosen_set):
                union_set = list(set(final_chosen_set).union(knum_chosen_set))
                check_set = np.setdiff1d(range(p),union_set)
                check_press1 =np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,final_chosen_set]) for i in check_set ])
                check_press2 = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]) for i in check_set ])
                # print(final_chosen_set,'check_press1 ',check_press1 ,knum_chosen_set,'check_press2',check_press2 )
                if check_press1<check_press2:
                    knum_chosen_set = final_chosen_set
                # else:
                # # regret to last knum chosen_set and ssr
                #     SSR = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]) for i in np.setdiff1d(range(p),knum_chosen_set)])
                #     final_chosen_set = knum_chosen_set 
                    
                # print('knum',knum,knum_chosen_set)
        return [knum_chosen_set ,final_svd_C]     
        # Zk, Theta_tru, svd_C_hand = compute_k_truncated_svd(Znorm, knum) 
        # hand_svd_C= svd_C_hand*.5
        
        # Gammavalue = svd_convex_optimization(Znorm, hand_svd_C, knum)[1]
        
        # hand_chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
        
        # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,p,Znorm)
        
        # return knum_chosen_set,SSR,final_svd_C,hand_chosen_set,hand_ssr,hand_svd_C
    elif method == 'adjDGL':
      
        chosen_set =[]
        # Znorm = Z
        knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        # knum_list = np.array([Knum])
        svd_C_list = np.linspace(svd_C*0.1, svd_C, 30)
        # svd_C_list = np.array([svd_C])

        SSR = 999
        final_knum = 0
        final_chosen_set = []
        knum_chosen_set = []
        cv_ssr = SSR-1
        final_svd_C  = 0
        for knum in knum_list:
            for svd_C_i in svd_C_list:
                if asset[0][0]:
                    asset = asset[:train_size]
                    asset = normalize_columns(asset)
                    asset_train,asset_test = train_test(asset,train_size*2//3)
                # St = sr_best(Znorm,lambda_val = 0.031, sigma = 500, T = 10000)
                Gammavalue = adjDGL_convex_optimization(Znorm, svd_C_i, knum)[1]
                
                nonzero = np.count_nonzero(np.round(cp.norm(Gammavalue, 2, axis=1).value,6))
                
                if nonzero == 0:
                    pass
                elif  0<nonzero<knum :
                    # chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm)
                elif nonzero >= knum:
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)

                if cv_ssr<SSR:
                    SSR = cv_ssr
                    final_chosen_set =  chosen_set
                   
                    final_knum = knum
                    final_svd_C = svd_C_i
            
   
            if  len(knum_chosen_set)==0:
                knum_chosen_set = final_chosen_set
            elif len(final_chosen_set) >= len(knum_chosen_set):
                union_set = list(set(final_chosen_set).union(knum_chosen_set))
                check_set = np.setdiff1d(range(p),union_set)
                check_press1 =np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,final_chosen_set]) for i in check_set ])
                check_press2 = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]) for i in check_set ])
                # print(final_chosen_set,'check_press1 ',check_press1 ,knum_chosen_set,'check_press2',check_press2 )
                if check_press1<check_press2:
                    knum_chosen_set = final_chosen_set
          
        
          # print('knum',knum,knum_chosen_set)
        return knum_chosen_set
    # elif method == 'adjSGL':
        
    #     chosen_set =[]
    #     # Znorm = Z
    #     knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
    #     # knum_list = np.array([Knum])
    #     # svd_C_list = np.linspace(svd_C*0.1, svd_C, 10)
    #     svd_C_list = np.array([svd_C])

    #     SSR = 999
    #     final_knum = 0
    #     final_chosen_set = []
    #     knum_chosen_set = []
    #     cv_ssr = SSR-1
    #     final_svd_C  = 0
    #     for knum in knum_list:
    #         for svd_C_i in svd_C_list:
    #             if asset[0][0]:
    #                 asset = asset[:train_size]
    #                 asset = normalize_columns(asset)
    #                 asset_train,asset_test = train_test(asset,train_size*2//3)
    #             # St = sr_best(Znorm,lambda_val = 0.031, sigma = 500, T = 10000)
    #             Gammavalue = adjSGL_convex_optimization(Znorm, svd_C_i, knum)[1]
                
    #             nonzero = np.count_nonzero(np.round(cp.norm(Gammavalue, 2, axis=1).value,6))
                
    #             if nonzero == 0:
    #                 pass
    #             elif  0<nonzero<knum :
    #                 # chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
    #                 chosen_set,cv_ssr= chosen_set_with_sr_press(Gammavalue,nonzero,Znorm)
    #             elif nonzero >= knum:
    #                 chosen_set,cv_ssr= chosen_set_with_sr_press(Gammavalue,knum,Znorm)

    #             if cv_ssr<SSR:
    #                 SSR = cv_ssr
    #                 final_chosen_set =  chosen_set
                    
    #                 final_knum = knum
    #                 final_svd_C = svd_C_i
            
    #         Z_hat = np.zeros_like(Znorm)
    #         alpha_hat = np.zeros(p)
    #         for i in range(p):      
            
    #             Z_minus_i = np.delete(Znorm, i, axis=1) 
    #             Ui,Si,Vi = np.linalg.svd(Z_minus_i)
    #             X = np.hstack((np.ones((n, 1)), Ui[:,:nonzero]))
    #             # X = np.hstack((np.ones((n, 1)),Z_minus_i ))
    #             beta = np.linalg.lstsq(X, Znorm[:, i], rcond=None)[0]
    #             alpha_hat[i]=beta[0]
    #             Z_hat[:, i] = np.dot(X, beta)
            
    #         Sigma = np.cov(Znorm - Z_hat, rowvar=False)
    #         S,U=np.linalg.eig(Sigma)

    #         Sigma_half = U@np.diag(np.sqrt(S))@U.T
        
    #         alpha_z_score = np.abs(alpha_hat) / np.diagonal(Sigma_half)
    #         if  len(knum_chosen_set)==0:
    #             knum_chosen_set = final_chosen_set
    #         elif len(final_chosen_set) >= len(knum_chosen_set):
    #             union_set = list(set(final_chosen_set).union(knum_chosen_set))
    #             check_set = np.setdiff1d(range(p),union_set)
                
    #             check_press1 =np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,final_chosen_set]@ np.diag(alpha_z_score[final_chosen_set])) for i in check_set ])
    #             check_press2 = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]@ np.diag(alpha_z_score[knum_chosen_set])) for i in check_set ])
    #             # print(final_chosen_set,'check_press1 ',check_press1 ,knum_chosen_set,'check_press2',check_press2 )
    #             if check_press1<check_press2:
    #                 knum_chosen_set = final_chosen_set
            
        
    #         # print('knum',knum,knum_chosen_set)
    #     return knum_chosen_set
    elif method == 'adjSGL':
        
        chosen_set =[]
        # Znorm = Z
        knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        # knum_list = np.array([Knum])
        svd_C_list = np.linspace(svd_C*0.1, svd_C, 10)
        # svd_C_list = np.array([svd_C])

        SSR = 999
        final_knum = 0
        final_chosen_set = []
        knum_chosen_set = []
        cv_ssr = SSR-1
        final_svd_C  = 0
        for knum in knum_list:
            for svd_C_i in svd_C_list:
                if asset[0][0]:
                    asset = asset[:train_size]
                    asset = normalize_columns(asset)
                    asset_train,asset_test = train_test(asset,train_size*2//3)
                # St = sr_best(Znorm,lambda_val = 0.031, sigma = 500, T = 10000)
                Gammavalue = adjSGL_convex_optimization(Znorm, svd_C_i, knum)[1]
                
                nonzero = np.count_nonzero(np.round(cp.norm(Gammavalue, 2, axis=1).value,6))
                
                if nonzero == 0:
                    pass
                elif  0<nonzero<knum :
                    # chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm)
                elif nonzero >= knum:
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)

                if cv_ssr<SSR:
                    SSR = cv_ssr
                    final_chosen_set =  chosen_set
                    
                    final_knum = knum
                    final_svd_C = svd_C_i
            

            if  len(knum_chosen_set)==0:
                knum_chosen_set = final_chosen_set
            elif len(final_chosen_set) >= len(knum_chosen_set):
                union_set = list(set(final_chosen_set).union(knum_chosen_set))
                check_set = np.setdiff1d(range(p),union_set)
                
                check_press1 =np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,final_chosen_set]) for i in check_set ])
                check_press2 = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]) for i in check_set ])
                # print(final_chosen_set,'check_press1 ',check_press1 ,knum_chosen_set,'check_press2',check_press2 )
                if check_press1<check_press2:
                    knum_chosen_set = final_chosen_set
            
        
            # print('knum',knum,knum_chosen_set)
        return knum_chosen_set

    elif method == 'DGL':
      
        chosen_set =[]
        # Znorm = Z
        # knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        knum_list = np.array([Knum])
        svd_C_list = np.linspace(svd_C*0.1, svd_C, 30)
        # svd_C_list = np.array([svd_C])

        SSR = 999
        final_knum = 0
        final_chosen_set = []
        knum_chosen_set = []
        cv_ssr = SSR-1
        final_svd_C  = 0
        for knum in knum_list:
            for svd_C_i in svd_C_list:
                if asset[0][0]:
                    asset = asset[:train_size]
                    asset = normalize_columns(asset)
                    asset_train,asset_test = train_test(asset,train_size*2//3)
                # St = sr_best(Znorm,lambda_val = 0.031, sigma = 500, T = 10000)
                Gammavalue = Z_convex_optimization(Znorm, svd_C_i, knum)[1]
                
                nonzero = np.count_nonzero(np.round(cp.norm(Gammavalue, 2, axis=1).value,6))
                
                if nonzero == 0:
                    pass
                elif  0<nonzero<knum :
                    # chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm_train,Znorm_test)
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,nonzero,Znorm)
                elif nonzero >= knum:
                    chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)

                if cv_ssr<SSR:
                    SSR = cv_ssr
                    final_chosen_set =  chosen_set
                    final_knum = knum
                    final_svd_C = svd_C_i
            
   
            if  len(knum_chosen_set)==0:
                knum_chosen_set = final_chosen_set
            elif len(final_chosen_set) >= len(knum_chosen_set):
                union_set = list(set(final_chosen_set).union(knum_chosen_set))
                check_set = np.setdiff1d(range(p),union_set)
                check_press1 =np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,final_chosen_set]) for i in check_set ])
                check_press2 = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,knum_chosen_set]) for i in check_set ])
                # print(final_chosen_set,'check_press1 ',check_press1 ,knum_chosen_set,'check_press2',check_press2 )
                if check_press1 < check_press2:
                    knum_chosen_set = final_chosen_set
       
        
          # print('knum',knum,knum_chosen_set)
        return [knum_chosen_set ,final_svd_C]
        # Zk, Theta_tru, svd_C_hand = compute_k_truncated_svd(Znorm, knum) 
        # hand_svd_C= svd_C_hand*.5
        # Gammavalue = Z_convex_optimization(Znorm, hand_svd_C, knum)[1]
        # hand_chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
        # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,p,Znorm)
        
        # return knum_chosen_set,SSR,final_svd_C,hand_chosen_set,hand_ssr,hand_svd_C
    elif method == 'TSMS': 
        
        for knum in range(2,p//2):  
            
            if asset[0][0]:
                asset = asset[:train_size]
                asset = normalize_columns(asset)
                U = np.linalg.svd(np.hstack((Z,asset)))[0][:,:knum]
            else:
                U = get_U(Z, knum)
            chosen_set =[]
            l2_distance = np.zeros(p)
            Bhat = np.linalg.lstsq(U[:,:knum],Z, rcond=None)[0]
            Mhat = U[:,:knum]@Bhat
            l2_distance = np.linalg.norm(Mhat-Z,axis = 0)
            if fix_true:
                l2_distance[fix_true] = np.zeros(len(fix_true))
            if fix_false:
                l2_distance[fix_false] = np.ones(len(fix_false))*100
            chosen_num = heapq.nsmallest(knum,l2_distance)
            
            
            for t in chosen_num:
                index = list(l2_distance).index(t)
                chosen_set.append(index)
                l2_distance[index]= -1 #give a value that it wont be chosen again
            # print(chosen_set) 
            Y = Z[:,list(set(range(p)).difference(chosen_set))]
            
            # p_value = PY_test(Z[:,chosen_set],Y,[[False]])[0]
            Xj = Z[:,chosen_set]
            Xi = Z[:,chosen_set[:-1]]
            ssri = round(sum(np.linalg.lstsq(Xi,Y, rcond=None)[1]),9)
            ssrj = round(sum(np.linalg.lstsq(Xj,Y, rcond=None)[1]),9)
            f_test = (ssri- ssrj)/ (ssrj/(n-knum))
            p_value = 1 - scipy.stats.f.cdf(f_test,1, n-knum)
            # print(chosen_set)
            if  p_value > hyper_p:
                mutistep_chosen = list(itertools.combinations(chosen_set, knum-1))
                for i in mutistep_chosen:
                    ssri_try = round(sum(np.linalg.lstsq(Z[:,i],Y, rcond=None)[1]),9)
                    if ssri_try <= ssri:
                        chosen_set = list(i)
                        ssri = ssri_try
            f_test = (ssri- ssrj)/ (ssrj/(n-knum))           
            p_value = 1 - scipy.stats.f.cdf(f_test,1, n-knum)
            
            if p_value  > hyper_p:
                
                return chosen_set
               
    # hand_ssr = np.mean([PRESS_statistic(Znorm[:,i],Znorm[:,[0,1,2,3,4]]) for i in np.setdiff1d(range(p),[0,1,2,3,4])])
    # print('[0,1,2,3,4]:', hand_ssr)
    
    


def evaluate(num_chosen_set,N_simulations,k,p):
    Select_knum = []
    CPlist=[]
    CFlist=[]
    TRlist=[]
    FRlist=[]
    F1list = [] 

    count = []
    select_knum = []
    for i in num_chosen_set:
        count += i
        select_knum.append(len(i))

    result = Counter(count)
    print(result)

    totally_correct_kum =0
    subset_knum = 0
    TR = 0
    FR = 0
    F1=0
  
    true_set = list(range(k))
    com_set = set(range(p))-set(true_set)
    for i in num_chosen_set:
    
        not_select = set(range(p))-set(i)
        if sorted(i) == true_set:      
            totally_correct_kum += 1
        elif set(true_set).issubset(set(i)):
            # if set(i).issubset(set(true_set)):
            subset_knum += 1
        TR += len(set(i)&set(true_set))/len(set(true_set))

        FR += len(set(i)&set(com_set))/len(set(com_set))
            
        F1 += 2*len(set(i)&set(true_set))/(2*len(set(i)&set(true_set))+len(set(i)&set(com_set))+ len(set(not_select &set(true_set))))
    Select_knum.append(select_knum) 
    CPlist.append(round(subset_knum/N_simulations+totally_correct_kum/N_simulations,6))
    CFlist.append(round(totally_correct_kum/N_simulations,6))
    TRlist.append(round(TR/N_simulations,6))
    FRlist.append(round(FR/N_simulations,6))
    F1list.append(round(F1/N_simulations,6))

    print('selected factor size:',round(np.mean(select_knum),6),'CP:',round(subset_knum/N_simulations+totally_correct_kum/N_simulations,6),'CF:',round(totally_correct_kum/N_simulations,6),'TR:',round(TR/N_simulations,6),'FR:',round(FR/N_simulations,6),'F1:',round(F1/N_simulations,6))
    
    return round(np.mean(select_knum),6),round(subset_knum/N_simulations+totally_correct_kum/N_simulations,6),round(totally_correct_kum/N_simulations,6),round(TR/N_simulations,6),round(FR/N_simulations,6),round(F1/N_simulations,6)


def Step_SR(X_full0, Lambda, Break, test_asset):
    T = X_full0.shape[0]
    N = X_full0.shape[1] 
    names = X_full0.columns
    S,t = [],[]
    X_full_subset = X_full0
    # pGRS = p_GRS = p_GRS1 = p_alpha = p_alpha_GRS = p_GRS_asset = qf0 = qf01 = p_PY = p_PY_asset = GRS = qf0 = NULL
    # p_GRS_asset100 = p_GRS_asset500 = p_PY_asset100 = p_PY_asset500 = NULL
    if Break:
        k = N-2
    else:
        k = N-1
        
    model_index = np.empty((k, k+3),dtype=object)
    GRS,qf0,p_alpha,p_GRS,p_GRS_asset1,p_GRS_asset2,p_PY,p_PY_asset1,p_PY_asset2 =[np.zeros(k,dtype=object) for _ in range(9)]
    test_asset = test_asset.values
    
    for i in range(k):
    
        a = np.where(np.isin(names, S))[0]
            
        X_full= X_full0.values
        if  a is None:
            X_full_subset = X_full
        else:
            X_full_subset =  np.delete(X_full, a, axis=1)
        N_subset = X_full_subset.shape[1]
        names_subset = names[np.where(~np.isin(names, S))[0]]
      
        GRS_k = np.empty((N_subset,1))
        t1 = np.empty((N_subset,1))

        for j in range(N_subset):
            Mj = np.union1d(names_subset[j], S)
   
       
            b = np.where(np.isin(names, Mj))[0]
            X1 = X_full[:,b]
            X2 = np.delete(X_full, b, axis=1)         
            #GRS statistic
            if len(b)==1:
                W1 = 1/np.cov(X1.T)
                mu1 = np.mean(X1)
                SR1 = 1 + mu1.T*W1*mu1
            else:
                W1 = np.linalg.pinv(np.cov(X1.T))
                mu1 = np.mean(X1, axis=0)
                SR1 = 1 + np.dot(np.dot(mu1.T, W1), mu1)
           
            
            F = np.concatenate((X1, X2), axis=1)
            W = np.linalg.pinv(np.cov(F.T))
            mu = np.mean(F, axis=0)
            SR = 1 + np.dot(np.dot(mu.T, W), mu)

            GRS_k[j] = ((T - N) / (N - i)) * (SR / SR1 - 1)
         
        #    
        a1 = names_subset[np.argmin(GRS_k)]
  
        t2 = t1[np.argmin(GRS_k)]
       
        if Break :
           
            if i:  
                X1 = X_full0[S].values
                X2 = X_full0[a1].values
                model = sm.OLS(X2, sm.add_constant(X1))
                p_alphai = model.fit().pvalues[0]  
            else:
                p_alphai = 0
            
            p_alpha[i]=p_alphai
            
            S.append(a1)
            model_index[i,:i+1] = S
            b1 = np.where(np.isin(names, S))[0]
            F1 = X_full[:,b1]
            F2 = np.delete(X_full, b, axis=1)
       
            ### GRS no test asset
            result1 = GRS_test(F1, F2, [[False]], Lambda)
            p_GRSi = result1[0]
            p_GRS[i]=p_GRSi
      
            ### PY no test asset
            result3 = PY_test(F1, F2, [[False]])
            p_PYi = result3[0]
            p_PY[i]=p_PYi
        
            if test_asset[0][0]:
                
                ### GRS 100 test asset
                test_asset100 = test_asset[:,:100]  
                result2 = GRS_test(F1, F2, test_asset100,Lambda)
                p_GRS_asset1i = result2[0]
                p_GRS_asset1[i]=p_GRS_asset1i
                
                
                ### GRS 285 test asset
                result2 = GRS_test(F1, F2, test_asset,Lambda)
                p_GRS_asset2i = result2[0]
                p_GRS_asset2[i]=p_GRS_asset2i
               

                ### PY 100 test asset
                result4 = PY_test(F1, F2, test_asset100)
                p_PY_asset1i=result4[0]
                p_PY_asset1[i]=p_PY_asset1i
                
                ### PY 285 test asset
                result4 = PY_test(F1, F2, test_asset)
     
                p_PY_asset2i=result4[0]
                p_PY_asset2[i]=p_PY_asset2i        
            
            p_value = [p_alphai, p_GRSi, p_GRS_asset1i,p_GRS_asset2i, p_PYi, p_PY_asset1i,p_PY_asset2i]
       
            if min(p_value) > Lambda and i > 1:
                break    
        else:
            t.append(t2)
            S.append(t1)
            model_index[i, :i] = S
            GRSi = GRS_k[np.where(GRS_k == np.min(GRS_k))]
            GRS[i]=GRSi
            qf0i = 1 - f.cdf(GRSi, N - i, T - N)
            qf0[i]=qf0i
            if len(S) == 1:
                X_SR = X_full0[S].values
                SR = np.mean(X_SR) / np.std(X_SR)
            else:
                X_SR = X_full0[S].values
                W1 = np.linalg.pinv(np.cov(F1))
                mu1 = np.mean(F1, axis=0)
                SR = np.sqrt(mu1 @ W1 @ mu1.T)
            
            model_index[i, (k + 1)] = GRSi
            model_index[i, (k + 2)] = qf0i
            model_index[i, (k + 3)] = SR * np.sqrt(12)
    if Break:
        
        P_all = np.column_stack((p_alpha, p_GRS, p_GRS_asset1, p_GRS_asset2, p_PY, p_PY_asset1, p_PY_asset2))
        model_result = np.empty_like(P_all,dtype=object)
        for z in range(P_all.shape[1]):
            m11 = np.where(P_all[:, z] > Lambda)[0]
          
            if len(m11) > 0 and z < 1:
                m1 = np.min(m11)
                # model_result[:(m1-1), z] = model_index[m1-1, :(m1-1)]
           
                model_result[:(m1), z] = model_index[m1, :m1]
            elif len(m11) > 0 and z > 0:
                m1 = np.min(m11)
                # model_result[:m1, z] = model_index[m1-1, :m1]
                model_result[:m1+1, z] = model_index[m1, :m1+1]
            else:
                model_result[:model_result.shape[0], z] = model_index[model_index.shape[0]-1, :model_result.shape[0]]
            result = model_result[:,-1]
        return result[result!=None]
        # return [model_result, P_all]
    else:
      
        S_full =  X_full0[S].values
        model_index11 = model_index[:i, np.concatenate((np.arange(0, i), [k+1, k+2, k+3]))]
        return [S_full,model_index11, GRS, qf0, S]

def KNS(X_full, alpha1):
    names0= X_full.columns
    X_full = X_full.values
    T, n = X_full.shape

    X = np.cov(X_full.T)
    y = np.mean(X_full, axis=0)

    Kfld = 5
    m1 = int(T / Kfld)

    # lambda_lasso = np.exp(np.linspace(-15, -16, num=50))
    lambda_lasso = np.exp(np.linspace(-15.2, -18, num=50))
    alpha0 = np.linspace(0.1, alpha1, num=10)
    cvm111 = np.empty((len(alpha0), 2))
    for a in range(len(alpha0)):
        alpha2 = alpha0[a]
        cvm1 = np.empty((Kfld, len(lambda_lasso)))
        for k in range(Kfld):
            X_train = X_full[np.setdiff1d(list(range(T)),list(range(m1*k, m1*(k+1))))]
     
            X_test = X_full[list(range(m1*k, m1*(k+1)))]
            X1 = np.cov( X_train.T)
            y1 = np.mean(X_train, axis=0)

            X2 = np.cov( X_test.T)
            y2 = np.mean( X_test, axis=0)
            for z in range(len(lambda_lasso)):
                elastic= linear_model.ElasticNet(alpha=alpha2,l1_ratio=lambda_lasso[z])
                elastic.fit(X1,y1) 
                y2_pred = elastic.predict(X2) 
                cvm1[k, z] = np.sum((y2 - y2_pred)**2)
          
        cvm11 = np.mean(cvm1, axis=0)
        best_lambda = lambda_lasso[np.argmax(cvm11 == np.min(cvm11))]

        cvm111[:, 0] = alpha2
        cvm111[:, 1] = best_lambda   
   
    cvm1111 = cvm111[:, 1]
    best_lambda_fin = np.argmax(cvm1111 == np.min(cvm1111))

    best_model = linear_model.ElasticNet(alpha=cvm111[best_lambda_fin, 0],l1_ratio=cvm111[best_lambda_fin, 1])
    best_model.fit(X,y)
    model_fin = names0[np.where(best_model.coef_[:-1] != 0)]   
    # return model_fin.values,cvm111[best_lambda_fin,0], cvm111[best_lambda_fin,1]
    return model_fin.values
           
def FGX(X_full, test_asset, alpha1):
    names0= X_full.columns
    X_full = X_full.values
    test_asset=test_asset.values
    T1,K = X_full.shape
    N_test = test_asset.shape[1]
    tmp1 = np.hstack((X_full, test_asset))
    tmp2 = np.cov(tmp1.T)
    cov_h = tmp2[K:(K + N_test), :K]
    ER = np.matrix(np.mean(test_asset, axis=0))
    var11 = np.diag(np.cov(X_full.T))
    aa = np.tile(var11, (N_test, 1))
    beta = cov_h / aa
    
    penalty = np.mean(beta**2, axis=0)
    penalty = penalty / np.mean(penalty)
    Kfld = 5
    m1 = int(T1 / Kfld)

    # lambda_lasso = np.exp(np.linspace(-7.5, -15, num=50))
    lambda_lasso = np.exp(np.linspace(-14.6, -18, num=50))
    cvm1 = np.zeros((Kfld, len(lambda_lasso)))
    lasso = LassoCV(alphas=lambda_lasso, cv=Kfld)
    for k in range(Kfld):
        X_train = X_full[np.setdiff1d(list(range(T1)),list(range(m1*k, m1*(k+1))))]
        X_test = X_full[list(range(m1*k, m1*(k+1)))]
        asset_train = test_asset[np.setdiff1d(list(range(T1)),list(range(m1*k, m1*(k+1))))]
        asset_test = test_asset[list(range(m1*k, m1*(k+1)))]
        
        tmp1 = np.hstack((X_train, asset_train))
        cov_train = np.cov(tmp1.T)[K:(K+N_test), :K]
        ER_train = np.matrix(np.mean(asset_train, axis=0))

        tmp2 = np.hstack((X_test, asset_test))
        cov_test = np.cov(tmp2.T)[K:(K+N_test), :K]
        ER_test = np.matrix(np.mean(asset_test, axis=0))
       
        X = cov_train@np.diag(penalty)
        y = ER_train.T
        
        for z in range(len(lambda_lasso)) :
            elastic= linear_model.ElasticNet(alpha=alpha1,l1_ratio=lambda_lasso[z])
            elastic.fit(X,y) 
            ER_pred = elastic.predict(cov_test.dot(np.diag(penalty)))
            cvm1[k, z] = np.sum((np.array(ER_test - ER_pred)**2))
        
    cvm11 = np.mean(cvm1, axis=0)

    best_lambda = lambda_lasso[np.argmax(cvm11 == np.min(cvm11))]
    X = cov_h@np.diag(penalty)
    y = ER.T
 
    best_model =linear_model.ElasticNet(alpha=alpha1, l1_ratio=best_lambda)
    best_model.fit(X, y)
    model_fin = names0[np.where(best_model.coef_[:-1] != 0)]

    return model_fin
      
    
    
def invest_metric(X_test,name1,F2,model1):
    
    AVG_2 = np.mean(F2) * 100
    SR_2 = np.mean(F2) / np.std(F2)
    SR_2 = np.sqrt(12) * SR_2
    # print( f"{np.mean(AVG_2):.2f}", f"{np.mean(SR_2):.2f}")
    if len(model1) == 1:
        alpha_CAPM_2 = alpha_FF5_2 = alpha_q5_2 = 0
        result_MVE = [f"{AVG_2:.2f}", f"{SR_2:.2f}", alpha_CAPM_2, alpha_FF5_2, alpha_q5_2]
    else:
        S_FF5 = ["MKTRF", "SMB", "HML", "RMW", "CMA"]
       
        X_FF5 = X_test[S_FF5]
        X_FF5 = sm.add_constant(X_FF5)
       
        lm1 = sm.OLS(F2, X_FF5.values).fit()
        
        alpha_FF5_2 = lm1.params[0] * 100
        p_value_FF5_2 = lm1.pvalues[0]
        
        if len(name1)==22:
            S_q5 = ["MKTRF", "SMB", "IA", "ROE", "REG"]
        else:
            S_q5 = ["MKTRF", "ME", "IA", "ROE", "REG"]
        X_q5 = X_test[S_q5]
        X_q5 = sm.add_constant(X_q5)
        lm1 = sm.OLS(F2, X_q5.values).fit()
        alpha_q5_2 = lm1.params[0] * 100
        p_value_q5_2 = lm1.pvalues[0]

        S_CAPM = ["MKTRF"]
        X_CAPM = X_test[S_CAPM]
        X_CAPM = sm.add_constant(X_CAPM)
        lm1 = sm.OLS(F2, X_CAPM.values).fit()
        alpha_CAPM_2 = lm1.params[0] * 100
        p_value_CAPM_2 = lm1.pvalues[0]

        alpha_CAPM_star = f"{alpha_CAPM_2:.2f}" + judge_star(p_value_CAPM_2)
        alpha_FF5_star = f"{alpha_FF5_2:.2f}" + judge_star(p_value_FF5_2)
        alpha_q5_star = f"{alpha_q5_2:.2f}" + judge_star(p_value_q5_2)
        result_MVE = [f"{AVG_2:.2f}", f"{SR_2:.2f}", alpha_CAPM_star, alpha_FF5_star, alpha_q5_star]
    return result_MVE
def invest_perform(X_full_all, model1, name1,T_train):
   
    T = X_full_all.shape[1]
    X1 = X_full_all[model1][: T_train].values
    X_test = X_full_all[T_train:]
    X_train = X_full_all[:T_train]
 
    if len(model1) == 1:
        F2_in = X1
        result_MVE_in = result_N_in = invest_metric(X_train, name1, F2_in, model1)
        F2_out = X_test[model1].valules
        result_MVE_out = result_N_out = invest_metric(X_test, name1, F2_out, model1)
    else:
        F2_N_in = np.mean(X1, axis=1)
        F2_N_out = np.mean(X_test[model1].values, axis=1)
        W1 = np.linalg.pinv(np.cov(X1.T))
        mu1 = np.mean(X1, axis=0)
     
        b = W1.dot(mu1)
        b1 = b / np.sum(b)
        F2_in = b1.dot(X1.T)
        F2_out = b1.dot(X_test[model1].T)

        if np.mean(F2_in) <= 0:
            F2_in = -F2_in
            F2_out = -F2_out
  
        result_MVE_in = invest_metric(X_train, name1, F2_in, model1)
        result_MVE_out = invest_metric(X_test, name1, F2_out, model1)
        result_N_in = invest_metric(X_train, name1, F2_N_in, model1)
        result_N_out = invest_metric(X_test, name1, F2_N_out, model1)
    return result_MVE_in, result_MVE_out, result_N_in, result_N_out     

def R2_calc(X_full_all, model1, test_asset1, T_train):
    T = test_asset1.shape[0]
    test_asset1=test_asset1.values
    model0 = ['MKTRF']
    MKT = X_full_all[model0].values

    X1 = MKT[:T_train]
    X1_out =MKT[T_train:]

    X2 = test_asset1[:T_train]
    # X2_out = test_asset1[(T - T_train):]

    ones = np.ones((T_train, 1))
    full_X = np.hstack((ones, X1))
    beta = np.round(np.linalg.pinv(full_X.T.dot(full_X)).dot(full_X.T).dot(X2),15)
    beta = beta[1:, :]

    MKT_xhat1 = X1.dot(beta)
    MKT_xhat1_out = X1_out.dot(beta)

    lambda_X = np.tile(np.mean(X1, axis=0),(T_train, 1))
    MKT_xhat2 = lambda_X.dot(beta)

    lambda_X_out = np.tile(np.mean(X1_out, axis=0), ((T - T_train), 1))
    MKT_xhat2_out = lambda_X_out.dot(beta)

    bar_X2 = np.mean(X2, axis=0)
    b = np.hstack((np.ones((beta.shape[1], 1)), beta.T))

    gamma = np.linalg.pinv(b.T.dot(b)).dot(b.T).dot(bar_X2)
    MKT_xhat3 = beta.T.dot(gamma[1:])
    X1 = X_full_all[model1][:T_train].values
    full_X = np.hstack((ones, X1))

    beta = np.round(np.linalg.pinv(full_X.T.dot(full_X)).dot(full_X.T).dot(X2),15)
    beta = beta[1:, :]
    b = np.hstack((np.ones((beta.shape[1], 1)), beta.T))

    if len(model1) == 1:
        beta = beta.T

    xhat1 = X1.dot(beta)
    lambda_X = np.tile(np.mean(X1, axis=0), (X1.shape[0], 1))
    xhat2 = lambda_X.dot(beta)
    bar_X2 = np.mean(X2, axis=0)
    gamma = np.linalg.pinv(b.T.dot(b)).dot(b.T).dot(bar_X2)

    xhat3 = beta.T.dot(gamma[1:])
    Total_R2 = 1 - np.sum((X2 - xhat1)**2) / np.sum((X2 - MKT_xhat1)**2)
    Predictive_R2 = 1 - np.sum((X2 - xhat2)**2) / np.sum((X2 - MKT_xhat2)**2)
    Pricing_R2 = 1 - np.sum((np.mean(X2 - xhat1, axis=0))**2) / np.sum((np.mean(X2 - MKT_xhat1, axis=0))**2)
    cross_R2 = 1 - np.sum((bar_X2 - xhat3)**2) / np.sum((bar_X2 - MKT_xhat3)**2)

    R2_in = np.array([Total_R2, Predictive_R2, Pricing_R2, cross_R2]) * 100


    asset_test1 = test_asset1[(T_train):]
    X_test1 = X_full_all[model1][(T_train):].values
    xhat1_out = X_test1.dot(beta)
    lambda_X_out = np.tile(np.mean(X_test1, axis=0), (X_test1.shape[0], 1))
    xhat2_out = lambda_X_out.dot(beta)

    Total_R2_out = 1 - np.sum((asset_test1 - xhat1_out)**2) / np.sum((asset_test1 - MKT_xhat1_out)**2)
    Predictive_R2_out = 1 - np.sum((asset_test1 - xhat2_out)**2) / np.sum((asset_test1 - MKT_xhat2_out)**2)
    Pricing_R2_out = 1 - np.sum((np.mean(asset_test1 - xhat1_out, axis=0))**2) / np.sum((np.mean(asset_test1 - MKT_xhat1_out, axis=0))**2)
    bar_X2 = np.mean(asset_test1, axis=0)
    cross_R2_out = 1 - np.sum((bar_X2 - xhat3)**2) / np.sum((bar_X2 - MKT_xhat3)**2)

    R2_out = np.array([Total_R2_out, Predictive_R2_out, Pricing_R2_out, cross_R2_out]) * 100


    return R2_in, R2_out

def evaluate1(model_fin,X_full):
    
    names0 = X_full.columns
    result_FSE = np.empty((len(names0), (len(model_fin)+2)),dtype='object')
    for m in range(len(model_fin)):
       
        FSE_data = X_full[model_fin[m]].values
        ones = np.ones((X_full.shape[0], 1))
        FSE_data = np.hstack((ones, FSE_data))
        names1 = np.setdiff1d(names0,model_fin[m] )
        Test_data = X_full[names1].values
        
        tmp =  np.empty((len(names0)),dtype='object')
        for z in range(Test_data.shape[1]):
            
            X2 = Test_data[:,z]
            bb = [i for i, name in enumerate(names0) if name == names1[z]]
           
            result_FSE[bb, 0] = np.mean(X2, axis=0) * 100
            result_FSE[bb, 1] = np.mean(X2) / np.std(X2) * np.sqrt(X2.shape[0])
            
            lm1 =sm.OLS(X2, sm.add_constant(FSE_data))
            lm1=lm1.fit()
            
            # result_FSE[bb, m+2] = lm1.params[0] * 100
            result_FSE[bb, m+2] = lm1.pvalues[0] * 100
          
            # tmp[bb] = str(round((lm1.params[0] * 100),3))+judge_star(lm1.pvalues[0])
            tmp[bb] = judge_star(lm1.pvalues[0])
        
        result_FSE[:, m+2] = tmp 
           
    return result_FSE

def evaluate2(model_fin,X_full_all, T_train,name1):
    invest_result = np.empty((len(model_fin)*2, 10),dtype=object)
    
    for zzz in range(len(model_fin)):
        model1 = model_fin[zzz]
        invest_result1 =invest_perform(X_full_all, model1, name1,T_train)
        ## 1/N
        
        invest_result[zzz, :5] = invest_result1[2]  # in-sample
        invest_result[zzz, 5:10] = invest_result1[3]  # out-of-sample
        
        ## MVE
        invest_result[zzz+len(model_fin), :5] = invest_result1[0]  # in-sample
        invest_result[zzz+len(model_fin), 5:10] = invest_result1[1]  # out-of-sample

    invest_result_df = pd.DataFrame(invest_result)
    invest_result_df.columns = ["AVG", "SR", "alpha_CAPM", "alpha_FF5", "alpha_q5", "AVG", "SR", "alpha_CAPM", "alpha_FF5", "alpha_q5"]
    # invest_result_df.index = ['grplasso', 'BS2017', 'GRS', 'FSE', 'FSE(285)', 'KNS2020', 'FGX2020', 'CAPM', 'FF5', 'q5']
    invest_result_df.index = ['adjDGL','ZJ','FGX','ff5','q5',
                              'adjDGL','ZJ','FGX','ff5','q5']
    return invest_result_df
    # files = 'result/Table5_cv'
    # store_csv1 = files + '.csv'

    # invest_result_df.to_csv(store_csv1, index=True)
    
    
    