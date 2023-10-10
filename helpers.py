'''
Author: Naixin && naixinguo2-c@my.cityu.edu.hk
Date: 2023-08-15 14:09:12
LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
LastEditTime: 2023-10-10 17:00:06
FilePath: /trylab/grp_journal/helpers.py
Description:

'''

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
import random
import heapq
import itertools
import pandas as pd
import warnings
import cvxpy as cp
warnings.filterwarnings('ignore')

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
    result = problem.solve()
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value
def Z_convex_optimization(Z, svd_C, knum, St):
    # Define the dimensions of the matrices
    n, p = Z.shape

    # Define the optimization variable Gamma
    # Gamma = cp.Variable((p, p))
    Z0 = Z[:,St]
    Gamma0 =np.random.normal(0.001,0.01,(p,p))
    Gamma0[St] = np.linalg.pinv(Z0.T@Z0)@Z0.T@Z
    
    Gamma = cp.Variable((p,p))
    Gamma.value = Gamma0
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
    result = problem.solve()
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value

def  Znew1_convex_optimization(Z, svd_C, knum, St):
    
    # Define the dimensions of the matrices
    n, p = Z.shape
    mu0 = np.mean(Z, axis=0)
    Sigma0 = np.cov(Z.T)
    # Define the optimization variable Gamma
    # Gamma = cp.Variable((p, p))
    Z0 = Z[:,St]
    Gamma0 =np.random.normal(0.001,0.01,(p,p))
    Gamma0[St] = np.linalg.pinv(Z0.T@Z0)@Z0.T@Z
    # Define the objective function
    #####################################################################################
    # objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    Gamma = cp.Variable((p,p))
    Gamma.value = Gamma0
    # Sigma = (Z-Z@Gamma0).T@(Z-Z@Gamma0)/n
    Sigma = np.cov((Z-Z@Gamma0).T)
      
    S,U=np.linalg.eig(Sigma)
    Sigma_half_inv =U@np.diag(np.sqrt(1/S))@U.T

    for i in range(1):
     
        objective = cp.Minimize(cp.norm( Sigma_half_inv @ Z.T - Sigma_half_inv @ Gamma.T @ Z.T  , 'fro')**2)
        # Define the constraint
        constraints = [
            cp.sum(cp.norm(Gamma, 2, axis=1))<=svd_C
        ]
        # Define the optimization problem
        problem = cp.Problem(objective, constraints)
        # Solve the optimization problem
   
        result = problem.solve(warm_start = True)
        # Sigma = (Z-Z@Gamma.value).T@(Z-Z@Gamma.value)/n

        Sigma = np.cov((Z-Z@Gamma.value).T)
        S,U=np.linalg.eig(Sigma)
        Sigma_half_inv =U@np.diag(np.sqrt(1/S))@U.T

        
        St0 = list(np.argsort(-np.round(cp.norm(Gamma.value, 2, axis=1).value,6))[:knum])
        print('sr',i,St0, sr(np.linalg.solve(Sigma0[St0][:, St0], mu0[St0]), mu0[St0], Sigma0[St0][:, St0]))
        print(np.linalg.norm(Gamma.value, axis=0))
    # Return the optimal value and the optimal Gamma
    return result, Gamma.value
def  Znew2_convex_optimization(Z, svd_C, knum, St):
    # Define the dimensions of the matrices
    n, p = Z.shape
    # Define the optimization variable Gamma
    # Gamma = cp.Variable((p, p))
    Z0 = Z[:,St]
    Gamma0 =np.random.normal(0.001,0.01,(p,p))
    Gamma0[St] = np.linalg.pinv(Z0.T@Z0)@Z0.T@Z
    # Define the objective function
    #####################################################################################
    # objective = cp.Minimize(cp.norm(U - Z @ Gamma, 'fro')**2)
    Gamma = cp.Variable((p,p))
    alpha = cp.Variable((n,p))
    Gamma.value = Gamma0
    # Sigma = np.cov((Z-Z@Gamma0).T)
    Sigma = (Z-Z@Gamma0).T@(Z-Z@Gamma0)/n
    S,U=np.linalg.eig(Sigma)
    Sigma_half_inv =U@np.diag(np.sqrt(1/S))@U.T

    for _ in range(1):    
       
        Sigma_inv = np.linalg.pinv(Sigma)
        objective = cp.Minimize(cp.norm(Sigma_half_inv @ Z.T - Sigma_half_inv @ Gamma.T @ Z.T  - Sigma_half_inv@alpha, 'fro')**2 + alpha @ Sigma_inv @ alpha.T)
        # Define the constraint
        constraints = [
            cp.sum(cp.norm(Gamma, 2, axis=1))<=svd_C
        ]
        # Define the optimization problem
        problem = cp.Problem(objective, constraints)
        # Solve the optimization problem
        result = problem.solve(warm_start=True)
        # Sigma = np.cov((Z-Z@Gamma.value-alpha.value).T)
        Sigma = (Z-Z@Gamma.value-alpha.value).T@(Z-Z@Gamma.value-alpha.value)/n
        S,U=np.linalg.eig(Sigma)
        Sigma_half_inv =U@np.diag(np.sqrt(1/S))@U.T

    # Return the optimal value and the optimal Gamma
    return result, Gamma.value

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
        # svd_C_list = [1]
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
        return knum_chosen_set      
        # Zk, Theta_tru, svd_C_hand = compute_k_truncated_svd(Znorm, knum) 
        # hand_svd_C= svd_C_hand*.5
        
        # Gammavalue = svd_convex_optimization(Znorm, hand_svd_C, knum)[1]
        
        # hand_chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
        
        # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,p,Znorm)
        
        # return knum_chosen_set,SSR,final_svd_C,hand_chosen_set,hand_ssr,hand_svd_C
    elif method == 'DGL_new1':
      
        chosen_set =[]
        # Znorm = Z
        # knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        knum_list = np.array([Knum])
        # svd_C_list = np.linspace(svd_C*0.1, svd_C, 30)
        svd_C_list = [svd_C]

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
                St = sr_best(Znorm,lambda_val = 0.025, sigma = 500, T = 10000)
                Gammavalue = Znew1_convex_optimization(Znorm, svd_C_i, knum,St)[1]
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
        # Zk, Theta_tru, svd_C_hand = compute_k_truncated_svd(Znorm, knum) 
        # hand_svd_C= svd_C_hand*.5
        # Gammavalue = Z_convex_optimization(Znorm, hand_svd_C, knum)[1]
        # hand_chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
        # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,p,Znorm)
        
        # return knum_chosen_set,SSR,final_svd_C,hand_chosen_set,hand_ssr,hand_svd_C
    elif method == 'DGL_new2':
      
        chosen_set =[]
        # Znorm = Z
        # knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        knum_list = np.array([Knum])
        svd_C_list = np.linspace(svd_C*0.1, svd_C, 30)

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
                St = sr_best(Znorm,lambda_val = 0.033, sigma = 500, T = 10000)
                Gammavalue = Znew2_convex_optimization(Znorm, svd_C_i, knum,St)[1]
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
        # Zk, Theta_tru, svd_C_hand = compute_k_truncated_svd(Znorm, knum) 
        # hand_svd_C= svd_C_hand*.5
        # Gammavalue = Z_convex_optimization(Znorm, hand_svd_C, knum)[1]
        # hand_chosen_set,hand_ssr= chosen_set_with_press(Gammavalue,knum,Znorm)
        # chosen_set,cv_ssr= chosen_set_with_press(Gammavalue,p,Znorm)
        
        # return knum_chosen_set,SSR,final_svd_C,hand_chosen_set,hand_ssr,hand_svd_C
 
    elif method == 'DGL':
      
        chosen_set =[]
        # Znorm = Z
        # knum_list = np.array([Knum-2,Knum-1,Knum,Knum+1,Knum+2])
        knum_list = np.array([Knum])
        # svd_C_list = np.linspace(svd_C*0.1, svd_C, 30)
        svd_C_list = np.array([svd_C])

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
                St = sr_best(Znorm,lambda_val = 0.031, sigma = 500, T = 10000)
                Gammavalue = Z_convex_optimization(Znorm, svd_C_i, knum,St)[1]
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
    
    


def evaluate(num_chosen_set,N_simulations):
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
    true_set = [0,1,2,3,4]
    com_set = set(range(25))-set(true_set)
    for i in num_chosen_set:
    
        not_select = set(range(25))-set(i)
        if sorted(i) == true_set:      
            totally_correct_kum += 1
        elif set(true_set).issubset(set(i)):
            # if set(i).issubset(set(true_set)):
            subset_knum += 1
        TR += len(set(i)&set(true_set))/len(set(true_set))

        FR += len(set(i)&set(com_set))/len(set(com_set))
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