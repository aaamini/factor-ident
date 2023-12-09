import numpy as np
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.20f}'.format})
from scipy.stats import norm
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from scipy.stats import f
from sklearn.preprocessing import normalize
import random
# pd.set_option('display.float_format', '{:.20f}'.format)
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.20f}'.format})
# sys.float_repr = lambda x: format(x, '.20f')
import warnings
from helpers import *

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

n = 300
plist = [25, 65, 105,145, 185,265] 
# knum = 4
knum = 5
N_simulations = 100
true_set = [0,1,2,3,4]

Chosen_svd_set_5 =[]

def Final(i,p):
   Unionset = []
   Z = pd.read_csv('/home/yuanzhi/grouplasso/try_ZJ_olddgp/generate_'+str(p-knum)+'_300/Z_'+str(i)+'.csv')
   Z = np.array(Z[Z.columns[1:]]).T
   unionset = choose_factor('DGL', Z, knum, train_size=300, svd_C = compute_svd_C(Z, knum), asset=[[None]],fix_true = None,fix_false = None,hyper_p = 0.1) 
#    unionset = choose_factor('DGL', Z, knum, train_size=600, svd_C = 5, asset=[[None]],fix_true = None,fix_false = None,hyper_p = 0.1) 
   return  unionset
Select_knum = []
CPlist=[]
CFlist=[]
TRlist=[]
FRlist=[]
F1list = [] 
for p in plist:
    N_chosen_set= Parallel(n_jobs=50)(delayed(Final)(i,p) for i in range(N_simulations))

    num_chosen_set=[]
    Final_knum=[]
    for unionset in N_chosen_set:
        num_chosen_set.append(unionset[0]) 
        Final_knum.append(len(unionset[0]))
        
    result=evaluate(num_chosen_set,N_simulations,knum,p) #3/5
    print(result)
    Select_knum.append(result[0]) 
    CPlist.append(result[1])
    CFlist.append(result[2])
    TRlist.append(result[3])
    FRlist.append(result[4])
    F1list.append(result[5])
    
fig = plt.figure(figsize =(4, 3))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
x = plist
plt.grid(linestyle="--", alpha=0.3)
plt.title('n = 300, k = 5')
plt.xlabel('number of candidate factors '+ '$ p$')
plt.ylabel('rate')  # y轴标题
plt.plot(x, np.array(CPlist), marker='o', markersize=3)  
plt.plot(x, np.array(CFlist), marker='o', markersize=3)
plt.plot(x, np.array(TRlist), marker='o', markersize=3)
plt.plot(x, np.array(FRlist), marker='o', markersize=3)


plt.legend(['CP', 'CF', 'TPR', 'FPR'],prop = {'size':8})  

plt.show()  # 显示折线图
fig.savefig('simu_result/plot_300_5_dgl_zj.pdf',dpi=600,format='pdf',bbox_inches="tight")



    

