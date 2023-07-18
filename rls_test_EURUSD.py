#%%
import numpy as np
import pandas as pd
from ewrls_rolling import EwRls
import matplotlib.pyplot as plt
from gau_rff import GaussianRFF
from scipy.stats import pearsonr
import time
import pandas_datareader as pdr

np.set_printoptions(suppress=True)

dta = pdr.get_data_yahoo('SPY')
dta = dta["Close"].to_numpy()
close = dta.reshape(1,len(dta))

#%%
#return formula (first element in array is oldest data)
ret = lambda x: (x[0,1:]- x[0,:-1])/ x[0,:-1]

#lag matrix creation
lag_matrix = lambda x,lag: np.array([[x[0,j+i] for j in range(lag)] for i in range(x.shape[1]-lag)])

#%%
np.random.seed(0)
D = 1000
lags = 14
exog = ret(close).reshape(1,close.shape[1]-1)
#exog = close.reshape(1,close.shape[1])
norm_exog = np.zeros((1,exog.shape[1]-1))

for i in range(2,exog.shape[1]):
    norm_exog[0,i-2] = (exog[0,i-1] - exog[0,:i].mean())/exog[0,:i].std()

endog = norm_exog[0,lags:].reshape(1,norm_exog.shape[1]-lags)

#%%
st = time.time()
norm_exog = lag_matrix(norm_exog,14).T

def rff_test(norm_exog, endog, D, sigma, n, ff, l, roll_size, batch_s):
    '''
    batch_s: batch size
    '''

    lags = norm_exog.shape[0]
    rff = GaussianRFF(lags,D,sigma)
    
    mod = EwRls(rff.transform(norm_exog[:,:batch_s].reshape(lags,batch_s)),endog[0,:batch_s],roll_size,ff,l)
    preds = [mod.pred(rff.transform(norm_exog[:,batch_s].reshape(lags,1)))]
    mse = [(preds[0]-endog[0,batch_s])**2]

    for i in range(1,n):
        u = rff.transform(norm_exog[:,batch_s+i-1].reshape(lags,1)) # reshape feature vector to (4,1)
        d = endog[0,batch_s+i-1]
        mod.update(u,d)
        preds.append(mod.pred(rff.transform(norm_exog[:,batch_s+i].reshape(lags,1))))
        mse.append((preds[i]-endog[0,batch_s+i])**2)

    return preds,mse


#%%
''' Test to measure the sigma
all_mse = []
D = 1000
for i in range(-2,6):

    preds,mse = rff_test(norm_exog, endog,D, 2**i, 30, 1, 0, 52, 50)
    all_mse.append(np.mean(mse))
    print(i)


        
#%%

plt.plot(all_mse)
'''

#%%
D = 20
n = 30 #252*3
roll = 15
batch = int(roll/2)
preds,mse = rff_test(norm_exog, endog, D, 1, n, 1, 0, roll,batch)

print(np.mean(mse))
et = time.time()
elapsed_time = et - st
print(elapsed_time)


plt.plot(endog[0,batch-1:n+batch-1].T)
plt.plot(preds)
plt.show()

# %%
corr = pearsonr(endog[0,batch-1:n+batch-1].T,preds)
print(corr)

# %%
