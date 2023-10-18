from helpers import  *
import numpy as np

import pandas as pd
import warnings


warnings.filterwarnings('ignore')
fama285_ = pd.read_csv('empirical_data/Fama285_22.csv')
fama285 = fama285_[fama285_.columns[1:]]
df99 = pd.read_csv('empirical_data/X99_22.csv')
dff99 = df99[df99.columns[1:]]

Z = dff99.values
Z = normalize_columns(Z)
T = 10000
sigma =0.00001
lambd1 = 2/sigma
k = 9
lambd2 = 4/sigma

n, p = Z.shape
Mh = compute_k_truncated_svd(Z, k)[0]  # Perform k-truncated SVD of Z

# Calculate sample mean and sample covariance matrix

# n, p = Z.shape
# U = np.zeros((n, k))
alpha_hat = np.zeros(p)
Z_hat = np.zeros_like(Z)
for i in range(p):
    Z_minus_i = np.delete(Z, i, axis=1)
    Ui,Si,Vi = np.linalg.svd(Z_minus_i)
    X = np.hstack((np.ones((n, 1)), Ui[:,:k]))
    beta = np.linalg.lstsq(X, Z[:, i], rcond=None)[0]
    alpha_hat[i]=beta[0]
    Z_hat[:, i] = np.dot(X, beta)
Sigma = np.cov(Z - Z_hat, rowvar=False)   


# alpha_hat = np.abs(np.mean(Z - Mh, axis=0))
# Sigma = np.cov((Z - Mh).T)
S,U=np.linalg.eig(Sigma)
S = np.abs(S)
Sigma_inv = np.linalg.inv(Sigma)
Sigma_half_inv =np.linalg.inv(U@np.diag(np.sqrt(S))@U.T)
# Initialize S1 with indices of top 2k largest values of alpha_i / sqrt(Sigma_ii)
indices = np.argsort(alpha_hat / np.sqrt(np.diag(Sigma)))[::-1][:30]
#print('initial', dff99.columns[indices])

St = set(indices)
e1 = np.eye(len(St)+1)[0]
Z_St = np.concatenate((np.ones((n,1)),Z[:, list(St)]),axis=1)
Theta = np.linalg.inv(Z_St.T @ Z_St+lambd1*e1@e1.T) @ Z_St.T  @ Z 
Objective = np.zeros(T)
aSa = np.zeros(T)
RSS = np.zeros(T)
#record the subset selection in each MH update
Sseq = np.zeros((p, T))


for t in range(T):
    e1 = np.eye(len(St)+1)[0]
    j = np.random.randint(p)  # Generate a random index
    # Create new subset S_new by adding or removing j
    
    if j in St:
        S_new = St - {j}
    else:
        S_new = St | {j}
        
    Z_S_new = np.concatenate((np.ones((n,1)),Z[:, list(S_new)]),axis=1)    
    Theta_new = np.linalg.inv(Z_S_new.T @ Z_S_new+lambd1*e1@e1.T) @ Z_S_new.T  @ Z 

    # Calculate the new Theta and acceptance probability
   
    # acceptance_prob = np.exp(
    #     -sigma * np.linalg.norm(( Sigma_half_inv @ Z.T) - ( Sigma_half_inv  @ (Z_S_new @ Theta_new).T), 'fro')**2
    #     - lambd * len(S_new)
    #     +sigma * np.linalg.norm(( Sigma_half_inv @ Z.T) - ( Sigma_half_inv @ (Z_St@ Theta).T), 'fro')**2
    #     + lambd * len(St)
    # )
    acceptance_prob = np.exp(sigma * (-np.linalg.norm(( Sigma_half_inv @ Z.T) - ( Sigma_half_inv  @ (Z_S_new @ Theta_new).T), 'fro')**2
        -lambd1 * Theta_new[0]@ Sigma_inv @Theta_new[0].T
        -lambd2 * len(S_new)
        + np.linalg.norm((Sigma_half_inv @ Z.T) - ( Sigma_half_inv @ (Z_St@ Theta).T), 'fro')**2
        +lambd1 * Theta[0]@ Sigma_inv @Theta[0].T
        + lambd2 * len(St)
    ))
    
    # Set the new subset S_t+1 with probability according to acceptance probability
    if np.random.rand() < acceptance_prob:
        St = S_new
        #print(St)
        Z_St = np.concatenate((np.ones((n,1)),Z[:, list(St)]),axis=1)
        Theta = np.linalg.inv(Z_St.T @ Z_St+lambd1*e1@e1.T) @ Z_St.T  @ Z 
    Sseq[list(S_new), t] = 1
    #record Sharpe ratio + penalty
    Objective[t] =  np.linalg.norm((Sigma_half_inv @ Z.T) - ( Sigma_half_inv @ (Z_St@ Theta).T), 'fro')**2+lambd1 * Theta[0]@ Sigma_inv @Theta[0].T + lambd2 * len(St)
    aSa[t] = Theta[0]@ Sigma_inv @Theta[0].T
    RSS[t] = np.linalg.norm((Sigma_half_inv @ Z.T) - ( Sigma_half_inv @ (Z_St@ Theta).T), 'fro')**2
import matplotlib.pyplot as plt
#Plots
plt.plot(Objective[1000:]/100000)
plt.plot(np.sum(Sseq[:,1000:], axis=0))    
sridx = np.argmin(Objective)
#Find the indices of selected factors
Sopt = np.where(Sseq[:, sridx])[0]
print('opt',dff99.columns[Sopt])

mu = np.mean(Z, axis=0)
Sigma = np.cov(Z.T)
wt = np.linalg.solve(Sigma[Sopt][:, Sopt], mu[Sopt])
print('sr',sr(wt, mu[Sopt], Sigma[Sopt][:, Sopt]))
print('objective', Objective[sridx])
print('aSa', lambd1*aSa[sridx])
print('RSS', RSS[sridx])