
from helpers import *

df100 = pd.read_csv('empirical_data/X100_22.csv')
dff100 = df100[df100.columns[1:]]
fama285_ = pd.read_csv('/home/guonaixin/trylab/factor-ident/empirical_data/Fama285_22.csv')
fama285 = fama285_[fama285_.columns[1:]]

Z = dff100.values
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
  

def adj(Method,Z,train_size,Test_asset,k, par,hypermin,hypermax,hypernum):
    #100 training -1


    Z = normalize_columns(Z)[:train_size]
    test_asset = normalize_columns(Test_asset)[:train_size]
    n, p = Z.shape
    Lambd2 = np.linspace(hypermin,hypermax, hypernum)
    Chosen_set = {}
    PYtest = {}
    SR = {}
    BKRS = {}
    GRS = {}
    
    for lambd2 in Lambd2:
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
        S,U=np.linalg.eig(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_half = U@np.diag(np.sqrt(S))@U.T
        Sigma_half_inv =np.linalg.inv(Sigma_half)
        alpha_z_score = np.abs(alpha_hat) / np.diagonal(Sigma_half)
           
            #Gamma = cp.Variable((p+1,p))

            # B = cp.Variable((p+1,p))
        if Method == 'AdjSGL':
            B = cp.Variable((p,k))
            ######################### OPT #####################################
            if np.array(Test_asset)[0][0] :
                U,S,V=np.linalg.svd(np.concatenate((Z, test_asset), axis=1))
            else:
                U,S,V=np.linalg.svd(Z)
            
            Opt_matrix1 = U[:,:k]
        elif Method == 'AdjDGL':
            B = cp.Variable((p,p))
            Opt_matrix1 =Z
            
        Opt_matrix2 = Z  
        objective = cp.Minimize(cp.norm(Opt_matrix1 - Opt_matrix2@ B, 'fro')**2 +lambd2*cp.sum(cp.norm(np.diag(alpha_z_score**par)@(B) , 2, axis=1)))
        
        problem = cp.Problem(objective)
        # Solve the optimization problem
        result = problem.solve(solver=cp.MOSEK)
        row_norms = np.linalg.norm((B.value), axis=1)  # Calculate the row norms
        nonzero = np.count_nonzero(np.round(row_norms,5))
        if nonzero ==0:
            break 
        if nonzero==1 :
            chosen_set = np.argsort(row_norms)[::-1][:nonzero]
            check_set = np.setdiff1d(range(p),chosen_set)
            # print(chosen_set,np.array(dff100.columns)[np.array(chosen_set)])
            mu = np.mean(Z, axis=0)
            Sigma = np.cov(Z.T)
            wt = np.linalg.solve(Sigma[chosen_set][:, chosen_set], mu[chosen_set])
            check_set = np.setdiff1d(range(p),chosen_set)
            
            if nonzero not in Chosen_set and nonzero<=10:
                Chosen_set[nonzero]= list(chosen_set)
                PYtest[nonzero]=PY_test(Z[:,chosen_set], Z[:,check_set], [[False]])
                SR[nonzero]=sr(wt, mu[chosen_set], Sigma[chosen_set][:, chosen_set])
                BKRS[nonzero]=BKRS_test(Z[:,chosen_set], Z[:,check_set])
                GRS[nonzero]=GRS_test(Z[:,chosen_set], Z[:,check_set], [[False]], 1)
            if nonzero in Chosen_set and nonzero<=10:
                if BKRS[nonzero][0] > BKRS_test(Z[:,chosen_set], Z[:,check_set])[0]:
                    Chosen_set[nonzero]= list(chosen_set)
                    SR[nonzero]=sr(wt, mu[chosen_set], Sigma[chosen_set][:, chosen_set])
                    PYtest[nonzero]  = PY_test(Z[:,chosen_set], Z[:,check_set], [[False]])
                    BKRS[nonzero]=BKRS_test(Z[:,chosen_set], Z[:,check_set])
                    GRS[nonzero]=GRS_test(Z[:,chosen_set], Z[:,check_set], [[False]], 1)
            break
        chosen_set = np.argsort(row_norms)[::-1][:nonzero]
        check_set = np.setdiff1d(range(p),chosen_set)
        # print(chosen_set,np.array(dff100.columns)[np.array(chosen_set)])
        mu = np.mean(Z, axis=0)
        Sigma = np.cov(Z.T)
        wt = np.linalg.solve(Sigma[chosen_set][:, chosen_set], mu[chosen_set])   
        
        if nonzero not in Chosen_set and nonzero<=10:
            Chosen_set[nonzero]= list(chosen_set)
            PYtest[nonzero]=PY_test(Z[:,chosen_set], Z[:,check_set], [[False]])
            SR[nonzero]=sr(wt, mu[chosen_set], Sigma[chosen_set][:, chosen_set])
            BKRS[nonzero]=BKRS_test(Z[:,chosen_set], Z[:,check_set])
            GRS[nonzero]=GRS_test(Z[:,chosen_set], Z[:,check_set], [[False]], 1)
        if nonzero in  Chosen_set and nonzero<=10:
            if BKRS[nonzero][0] > BKRS_test(Z[:,chosen_set], Z[:,check_set])[0]:
                Chosen_set[nonzero]= list(chosen_set)
                SR[nonzero]=sr(wt, mu[chosen_set], Sigma[chosen_set][:, chosen_set])
                PYtest[nonzero]  = PY_test(Z[:,chosen_set], Z[:,check_set], [[False]])
                BKRS[nonzero]=BKRS_test(Z[:,chosen_set], Z[:,check_set])
                GRS[nonzero]=GRS_test(Z[:,chosen_set], Z[:,check_set], [[False]], 1)
        
    return Chosen_set, SR, PYtest, BKRS, GRS, k


Klist = [1,2,3,4,5,6,7,8,9,10]


Result = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,36*12,[[False]],j, -2 ,0.01,2,5000) for j in Klist)
print('AdjSGL2',Result)


Result_all  = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,Z.shape[0],[[False]],j, -2 ,0.01,2,5000) for j in Klist)
print('AdjSGL2_all',Result_all) 
   

Result1= Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,36*12,[[False]],j, -1 ,0.07,2,5000) for j in Klist)
print('AdjSGL1',Result1) 
   

Result1_all= Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,Z.shape[0],[[False]],j, -1 ,0.01,2,5000) for j in Klist)
print('AdjSGL1_all',Result1_all) 
   
   
Result_asset  = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,36*12,fama285,j, -2 ,0.001,2,5000) for j in Klist)
print('AdjSGL2_asset',Result_asset) 


Result_asset1  = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,36*12,fama285,j, -1 ,0.01,2,2500) for j in Klist)
print('AdjSGL1_asset',Result_asset1) 

Result_all_asset = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,Z.shape[0],fama285,j, -2 ,0.001,2,5000) for j in Klist)
print('AdjSGL2_asset_all',Result_all_asset) 


Result_all_asset1 = Parallel(n_jobs=10)(delayed(adj)('AdjSGL',Z,Z.shape[0],fama285,j, -1 ,0.01,2,5000) for j in Klist)
print('AdjSGL1_asset_all',Result_all_asset1) 




Result2= Parallel(n_jobs=10)(delayed(adj)('AdjDGL',Z,36*12,[[False]],j, -2 ,0.09,4,2500) for j in Klist)
print('AdjDGL2',Result2) 


Result2_all= Parallel(n_jobs=10)(delayed(adj)('AdjDGL',Z,Z.shape[0],[[False]],j, -2 ,0.05,4,2500) for j in Klist)
print('AdjDGL2_all',Result2_all)

Result3= Parallel(n_jobs=10)(delayed(adj)('AdjDGL',Z,36*12,[[False]],j, -1 ,0.5,4,2500) for j in Klist)
print('AdjDGL1',Result3)

Result3_all= Parallel(n_jobs=10)(delayed(adj)('AdjDGL',Z,Z.shape[0],[[False]],j, -1 ,0.5,4,2500) for j in Klist)
print('AdjDGL1_all',Result3_all)