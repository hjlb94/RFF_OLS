#%%
import numpy as np
import pandas as pd
from new_QR_ewrls import EwRls
import matplotlib.pyplot as plt
from gau_rff import GaussianRFF
from scipy.stats import pearsonr
import time
import pandas_datareader as pdr
import plotly.graph_objects as go
import plotly.io as pio

np.set_printoptions(suppress=True)

dta = pdr.get_data_yahoo('SPY')
dta_1 = dta["Close"].to_numpy()
close = dta_1.reshape(1,len(dta_1))

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
def rff_train(norm_exog, endog, D, sigma, n, ff, l, roll_size, batch_s):
    '''
    batch_s: batch size
    '''

    lags = norm_exog.shape[0]
    rff = GaussianRFF(lags,D,sigma)
    
    mod = EwRls(rff.transform(norm_exog[:,:batch_s].reshape(lags,batch_s)),endog[0,:batch_s],roll_size,ff,l)
    preds = [mod.pred(rff.transform(norm_exog[:,batch_s-1].reshape(lags,1)))]
    mse = [(preds[0]-endog[0,batch_s])**2]

    for i in range(1,n):
        u = rff.transform(norm_exog[:,batch_s+i-1].reshape(lags,1)) # reshape feature vector to (4,1)
        d = endog[0,batch_s+i-1]
        mod.update(u,d)
        preds.append(mod.pred(rff.transform(norm_exog[:,batch_s+i-1].reshape(lags,1))))
        mse.append((preds[i]-endog[0,batch_s+i-1])**2)

    return preds,mse

#%% Test to measure the sigma

all_mse = []
D = lags
for i in range(2,11):

    preds,mse = rff_test(norm_exog, endog,D+2**i, 1, 100, 1, 0, 52, 40)
    all_mse.append(np.mean(mse))

#%%
all_mse_train = []
D = lags
for i in range(2,11):

    preds,mse = rff_train(norm_exog, endog,D+2**i, 1, 100, 1, 0, 52, 40)
    all_mse_train.append(np.mean(mse))



# %%
x = [D+2**i for i in range(0,11)]
fig = go.Figure()
fig.add_trace(go.Scatter(x = x , y= all_mse_train))
#x_new = np.append(x[0],x[5],x[-1])
#x_name = [str(i) for i in x_new]
fig.update_layout(title = "SPY 20/09/2017 - 04/12/2018" , yaxis_title = "MSE Train",xaxis_title = "Number of RFF")
'''fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = x_new,
        ticktext = x_name
    )
)'''
fig.update_xaxes(tickangle=0,tick0 =10)
pio.write_image(fig, 'DD_Train.png')

# %%
x = [D+2**i for i in range(2,11)]
fig = go.Figure()
x_new = np.log(x)
x_name = [str(D+2**i) for i in range(2,11)]
y_new = np.log(all_mse)
fig.add_trace(go.Scatter(x = x_new , y= y_new, name = "MSE Test"))
fig.update_layout(title = "SPY 20/09/2017 - 04/12/2018" , yaxis_title = "MSE Test",xaxis_title = "Number of RFF")
x_new[1] = None
y_name = [str(D+2**i) for i in range(2,11)]

#fig.update_xaxes(type="log")
#fig.update_yaxes(type="log")
#fig.update_layout(xaxis_type="log")
y_name = [str(round(i)) for i in all_mse]
y_new[5:7] = None
y_new[1] = None
y_new[-2] = None
fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = y_new,
        ticktext = y_name
    )
)
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = x_new,
        ticktext = x_name
    )
)
fig.update_xaxes(tickangle=0,tick0 =10)
pio.write_image(fig, 'DD_Test.png')

# %%

et = time.time()
elapsed_time = et - st
print(elapsed_time)