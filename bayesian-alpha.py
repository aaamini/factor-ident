import numpy as np
import pandas as pd


#reading dataset
df96 = pd.read_csv('empirical_data/X_full.csv')
X = df96[df96.columns[1:]]

X = X / np.sqrt(np.sum(X**2, axis=0))
R = np.corrcoef(X.T)
idx = np.argsort(np.sum(np.abs(R), axis=0))[::-1]
St = idx[:20]
T = 10000
RSSwPenalty = np.zeros(T)
Sseq = np.zeros((96, T))
lambda_val = 0.45
sigma = 200

def alpha(X, Y, S):
    X_subset = X.iloc[:, S]
    X_array = X_subset.values
    Y_array = Y.values
    Xt = X_array.T
    XtX = np.dot(Xt, X_array)
    XtY = np.dot(Xt, Y_array)
    b = np.linalg.solve(XtX, XtY)
    Y_pred = X_array@b
    residuals = Y_array - Y_pred
    mean = np.mean(residuals, axis=0)
    std = np.std(residuals, axis=0)
    t_score =  mean / std 
    mask = np.ones_like(t_score, dtype=bool)
    mask[S] = False
    abs_sum_t = np.sum(np.abs(t_score[mask]))
    return abs_sum_t

for t in range(T):
    i = np.random.randint(96)
    Snew = np.setdiff1d(np.union1d(i, St), np.intersect1d(i, St))
    llhratio = np.exp(-sigma * (alpha(X, X, Snew) - alpha(X, X, St) + lambda_val * (len(Snew) - len(St))))
    random_num = np.random.rand()
    if llhratio > random_num:
        St = Snew
    Sseq[Snew, t] = 1
    RSSwPenalty[t] = -alpha(X, X, Snew) - lambda_val * len(Snew)

import matplotlib.pyplot as plt

plt.plot(RSSwPenalty)
plt.plot(np.sum(Sseq, axis=0))
sridx = np.argmax(RSSwPenalty)
Sopt = np.where(Sseq[:, sridx])[0]
column_name = X.columns[Sopt]
print("Selected factors:", column_name)
print("sum of absolute t-score of the selected variables:", alpha(X,X,Sopt))
Ssort = idx[:len(Sopt)]
print("sum of absolute t-score of greedy selection with the same size as our selection:", alpha(X,X,Ssort))