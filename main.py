# %%
## COPY OF THE double_Descent_ORLS_EWRLS script for the moment
import os
import numpy as np
import pandas as pd
from ewrls_rolling import EwRls
#import matplotlib.pyplot as plt
from gau_rff import GaussianRFF
#from scipy.stats import pearsonr
import time
import plotly.graph_objects as go
import plotly.io as pio

np.set_printoptions(suppress=True)

nme = os.getcwd()
print(nme)

dta = pd.read_csv('/Users/hugobarette/Documents/Uni/Summer_project/Code_and_data/eurusd.csv')
#dta = pd.read_csv('data/fx/eurusd.csv')
dta = dta["close"].to_numpy()
dta = dta[:100000]
close = dta.reshape(1, len(dta))

##this section creates the close array, 100000 bits long

# %%
# return formula (first element in array is oldest data)
ret = lambda x: (x[0, 1:] - x[0, :-1]) / x[0, :-1]

# lag matrix creation
lag_matrix = lambda x, lag: np.array([[x[0, j + i] for j in range(lag)] for i in range(x.shape[1] - lag)])

## these don't produce any variables, are they just for later??

# %%
np.random.seed(0)
D = 1000
lags = 14
exog = ret(close).reshape(1, close.shape[1] - 1) #this seems to use the returns calculation on the close array
#exog is the returns
# exog = close.reshape(1,close.shape[1])
norm_exog = np.zeros((1, exog.shape[1] - 1))
#creates an array of zeros

for i in range(2, exog.shape[1]):
    norm_exog[0, i - 2] = (exog[0, i - 1] - exog[0, :i].mean()) / exog[0, :i].std()
#normalised returns is in norm_exog

endog = norm_exog[0, lags:].reshape(1, norm_exog.shape[1] - lags)
#not entirely sure what this is for? THESE ARE THE OBSERVATIONS---why are they shaped this way

# %%
st = time.time()
norm_exog = lag_matrix(norm_exog, 14).T
#creating the transpose of the lag matrix of the normed returns, this is the feature dataset!!


def ORLS(norm_exog, endog, D, sigma, n, ff, l, roll_size, batch_s):

    '''
    batch_s: batch size
    '''

    lags = norm_exog.shape[0]
    rff = GaussianRFF(lags, D, sigma) #this calls the gaussianRFF

    mod = EwRls(rff.transform(norm_exog[:, :batch_s].reshape(lags, batch_s)), endog[0, :batch_s], roll_size, ff, l)
    #calls EwRLS from ewrls_rolling - creates mod variable of class EwRLS

    preds = [mod.pred(rff.transform(norm_exog[:, batch_s - 1].reshape(lags, 1)))]
    #uses the pred method from the EwRLS class to create the predictions
    mse = [(preds[0] - endog[0, batch_s]) ** 2]

    for i in range(1, n):
        u = rff.transform(norm_exog[:, batch_s + i - 1].reshape(lags, 1))  # reshape feature vector to (4,1)
        d = endog[0, batch_s + i - 1]
        mod.update(u, d)
        preds.append(mod.pred(rff.transform(norm_exog[:, batch_s + i - 1].reshape(lags, 1))))
        mse.append((preds[i] - endog[0, batch_s + i - 1]) ** 2)

    return preds, mse

## this whole section is not relevant for me as I only need the EWRLS
# %%
def EWRLS(norm_exog, endog, D, sigma, n, ff, l, roll_size, batch_s, span):
    #span is the only new term here - as expected
    '''
    batch_s: batch size
    '''
    lags = norm_exog.shape[0]
    rff = GaussianRFF(lags, D, sigma)

    mod = EwRls(rff.transform(norm_exog[:, :batch_s].reshape(lags, batch_s)), endog[0, :batch_s], roll_size, ff, l,
                span)
            #inclusion of span means the forgetting factor is set at 1 - 2/(span +1)

    preds = [mod.pred(rff.transform(norm_exog[:, batch_s - 1].reshape(lags, 1)))]
    #classic predictions array
    mse = [(preds[0] - endog[0, batch_s]) ** 2]
    #computing the mse between the predictions and observations

    for i in range(1, n):
        u = rff.transform(norm_exog[:, batch_s + i - 1].reshape(lags, 1))  # reshape feature vector to (4,1)
        d = endog[0, batch_s + i - 1]
        mod.update(u, d)
        preds.append(mod.pred(rff.transform(norm_exog[:, batch_s + i - 1].reshape(lags, 1))))
        mse.append((preds[i] - endog[0, batch_s + i - 1]) ** 2)

    return preds, mse


# %% Test to measure the sigma

all_mse_ORLS = []
D = lags
window = 300
batch = int(window / 2)

for i in range(2, 14):
    preds, mse = ORLS(norm_exog, endog, D + 2 ** i, 1, 20, 1, 0, window, batch)
    all_mse_ORLS.append(np.mean(mse))

# %%
all_mse_EWRLS = []
D = lags
span = 300
window = 10000
batch = 1000

for i in range(2, 14):
    preds, mse = EWRLS(norm_exog, endog, D + 2 ** i, 1, 20, 1, 0, window, batch, span)
    all_mse_EWRLS.append(np.mean(mse))

# %%
x = [D + 2 ** i for i in range(0, 14)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=all_mse_ORLS))
# x_new = np.append(x[0],x[5],x[-1])
# x_name = [str(i) for i in x_new]
fig.update_layout(title="EURUSD ORLS", yaxis_title="MSE ORLS", xaxis_title="Number of RFF")
'''fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = x_new,
        ticktext = x_name
    )
)'''
fig.update_xaxes(tickangle=0, tick0=10)
pio.write_image(fig, 'DD_ORLS.png')

# %%
x = [D + 2 ** i for i in range(0, 11)]
fig = go.Figure()
x_new = np.log(x)
y_new = np.log(all_mse_EWRLS)
y_new = y_new[5:]
x_new = x_new[4:]
fig.add_trace(go.Scatter(x=x_new, y=y_new, name="MSE EWRLS"))
fig.update_layout(title="EURUSD EWRLS", yaxis_title="MSE EWRLS", xaxis_title="Number of RFF")

y_name = all_mse_EWRLS[5:]
x_name = [str(D + 2 ** i) for i in range(5, 14)]
y_name = [str(round(i, 3)) for i in y_name]
y_name[0] = ''
y_name[1] = ''
y_name[-1] = ''

fig.update_layout(
    yaxis=dict(
        tickmode='array',
        tickvals=y_new,
        ticktext=y_name
    )
)
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=x_new,
        ticktext=x_name
    )
)
# fig.update_xaxes(tickangle=0,tick0 =10)
pio.write_image(fig, 'DD_EWRLS.png')

# %%

et = time.time()
elapsed_time = et - st
print(elapsed_time)

