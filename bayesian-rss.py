import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#reading dataset
df96 = pd.read_csv('empirical_data/X_full.csv')
X = df96[df96.columns[1:]]

X = X / np.sqrt(np.sum(X**2, axis=0))
R = np.corrcoef(X.T)
idx = np.argsort(np.sum(np.abs(R), axis=0))[::-1]
St = idx[:20]
T = 10
RSSwPenalty = np.zeros(T)
Sseq = np.zeros((96, T))
lambda_val = 2.5
sigma = 10

def RSS(X, Y, S):
    X_subset = X.iloc[:, S]
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_subset, Y)
    Y_pred = model.predict(X_subset)
    residuals = Y - Y_pred
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
RSS(X, X, Sopt)
Ssort = idx[:len(Sopt)]
RSS(X, X, Ssort)