import numpy as np
import pandas as pd

#reading dataset
df96 = pd.read_csv('empirical_data/X_full.csv')
X = df96[df96.columns[1:]]
#calculate sample mean and sample covariance matrix
mu = np.mean(X, axis=0)
Sigma = np.cov(X.T)
#solve the optimal MVE weight for all factors
wall = np.linalg.solve(Sigma, mu)
wall = wall / np.linalg.norm(wall)
idx = np.argsort(wall)[::-1]
#sort factors, from important for SR to less important
Ssort = idx[:96]
#select first 20 as initialization
St = idx[:20]
#number of MH updates
T = 10000
#record the value of sharpe ratio + penalty in each MH update
SRwPenalty = np.zeros(T)
#record the subset selection in each MH update
Sseq = np.zeros((96, T))
#An very important hyperparameter. Roughly speaking, if the variable can improve the SR by this value, then I can be selected.
lambda_val = 0.06
#If sigma is large, then the MH updates are more stable. 
sigma = 200

#define the sharpe ratio
def sr(wt, mu, Sigma):
    return (np.dot(wt, mu) ** 2) / np.dot(np.dot(wt, Sigma), wt)

for t in range(T):
    #randomly select a variable to update
    i = np.random.randint(96)
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

import matplotlib.pyplot as plt

#Plots
plt.plot(SRwPenalty)
plt.plot(np.sum(Sseq, axis=0))
#Find the index with largest value of Sharpe ratio + penalty
sridx = np.argmax(SRwPenalty)
#Find the indices of selected factors
Sopt = np.where(Sseq[:, sridx])[0]
#Find the weight corrsponding to Sopt
wSopt = np.linalg.solve(Sigma[Sopt][:, Sopt], mu[Sopt])
column_name = X.columns[Sopt]
print("Selected factors:", column_name)
print("Sharpe ratio of the selected variables:", sr(wSopt, mu[Sopt], Sigma[Sopt][:, Sopt]))
Ssort = idx[:len(Sopt)]
wB = np.linalg.solve(Sigma[Ssort][:, Ssort], mu[Ssort])
print("Sharpe ratio by greedy selection with the same size as our selection:", sr(wB, mu[Ssort], Sigma[Ssort][:, Ssort]))

