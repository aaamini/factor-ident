'''
Author: Naixin && naixinguo2-c@my.cityu.edu.hk
Date: 2023-09-06 22:48:49
LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
LastEditTime: 2023-09-07 00:18:42
FilePath: /trylab/factor-ident/bayesian-rss.py
Description: 

'''
import numpy as np
import pandas as pd


#reading dataset
df96 = pd.read_csv('empirical_data/X_full.csv')
X = df96[df96.columns[1:]]

X = X / np.sqrt(np.sum(X**2, axis=0))
R = np.corrcoef(X.T)
idx = np.argsort(np.sum(np.abs(R), axis=0))[::-1]
St = idx[:20]
T = 1000000
RSSwPenalty = np.zeros(T)
Sseq = np.zeros((96, T))
lambda_val = 2
sigma = 10

def RSS(X, Y, S):
    X_subset = X.iloc[:, S]
    X_array = X_subset.values
    Y_array = Y.values
    Xt = X_array.T
    XtX = np.dot(Xt, X_array)
    XtY = np.dot(Xt, Y_array)
    b = np.linalg.solve(XtX, XtY)
    Y_pred = X_subset@b
    residuals = Y_array - Y_pred
    rss = np.sum(np.sum(residuals**2))
    return rss

for t in range(T):
    i = np.random.randint(96)
    Snew = np.setdiff1d(np.union1d(i, St), np.intersect1d(i, St))
    llhratio = np.exp(-sigma * (RSS(X, X, Snew) - RSS(X, X, St) + lambda_val * (len(Snew) - len(St))))
    random_num = np.random.rand()
    if llhratio > random_num:
        St = Snew
    Sseq[Snew, t] = 1
    RSSwPenalty[t] = -RSS(X, X, Snew) - lambda_val * len(Snew)

import matplotlib.pyplot as plt

plt.plot(RSSwPenalty)
plt.plot(np.sum(Sseq, axis=0))
sridx = np.argmax(RSSwPenalty)
Sopt = np.where(Sseq[:, sridx])[0]
column_name = X.columns[Sopt]
print("Selected factors:", column_name)
print("RSS of the selected variables:", RSS(X,X,Sopt))
Ssort = idx[:len(Sopt)]
print("RSS by greedy selection with the same size as our selection:", RSS(X,X,Ssort))