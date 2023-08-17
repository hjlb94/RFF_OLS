## THIS IS THE FILE I AM RUNNING ANF TESTING

import kaleido
import pandas as pd
import numpy as np
from ewrls_rolling import EwRls
#import matplotlib.pyplot as plt
from gau_rff import GaussianRFF
from DD_QR_ewrls import QrRlS
#from scipy.stats import pearsonr
import time
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

print('what the hell is going on?')

dta = pd.read_csv('/Users/hugobarette/Documents/Uni/Summer_project/Code_and_data/eurusd.csv')

dta2 = pd.read_csv('/Users/hugobarette/Documents/Uni/Summer_project/InflationData/fredmd_transf_df.csv')
dta2_A = dta2["RPI"].to_numpy()
#dta2_A = dta2_A[:100000]
rpi = dta2_A.reshape(1, len(dta2))

#lambda function to calc the returns
ret = lambda x : (x[0, 1:] - x[0, :-1]) / x[0, :-1]

exog = ret(rpi).reshape(1, rpi.shape[1] - 1) #these are the results, not using the same stuff as previous years wrt the lag matrix

#initialise normalised array of correct size
#actually that isn't relevant???????
norm_exog = np.zeros((1, exog.shape[1] -1)) #why does this require two brackets??

for i in range(2, exog.shape[1]):
    norm_exog[0, i-2] = (exog[0, i-1] - exog[0, :i].mean())/exog[0, :i].std()
#have normalised observations now.... the normalisation is rolling
endog = norm_exog #this is the observations array used in the EWRLS

#next we need the features dataset
dta2 = dta2.drop(columns = ['RPI', 'date']) #drop unnecessary columns

norm_dta2 = np.zeros((dta2.shape[0], dta2.shape[1] -1))
norm_dta2 = pd.DataFrame(norm_dta2)

#creates the normalised features matrix
for n in range(0, 125):
    for i in range(2, dta2.shape[0]):
        #iloc is row, column
        norm_dta2[n][i-2] = (dta2.iloc[i-1, n] - dta2.iloc[:i, n].mean())/dta2.iloc[:i,n].std()

norm_dta2 = norm_dta2.fillna(0)
norm_dta2 = np.array(norm_dta2.T) #transpose very important

#first section test
print('data imported and normalised')

## define the EWRLs
D = 1000 #row length of the RFF features output
lags = norm_dta2.shape[0] #this is a problem piece
span = 300
window = 50
batch = 10 #this is the column number of the RFF feature output
sigma = 1
rounds = 10

# %%
def EWRLS(norm_dta2, endog, D, sigma, n, ff, l, roll_size, batch_s, span): #defining an EWRLS function to call
    #span is the only new term here - as expected
    '''
    batch_s: batch size
    '''
    lags = norm_dta2.shape[0]
    rff = GaussianRFF(lags, D, sigma) #creating a method called RFF

    # mod = EwRls(rff.transform(norm_dta2[:, :batch_s].reshape(lags, batch_s)),  # x this is the RFF transform of the features dataset - using rff method
    #             endog[0, :batch_s], # y - this is the observations dataset
    #             roll_size, # max_obs - normal variable
    #             ff, #forgetting factor
    #             l, # this is lambda but is always equal to zero for us
    #             span) #span is about calculating the correct forgetting factor
    #         #inclusion of span means the forgetting factor is set at 1 - 2/(span +1)

    mod = QrRlS(rff.transform(norm_dta2[:, :batch_s].reshape(lags, batch_s)),  # x this is the RFF transform of the features dataset - using rff method
                endog[0, :batch_s],  # y - this is the observations dataset
                roll_size,  # max_obs - normal variable
                ff,  # forgetting factor
                l)  # this is lambda but is always equal to zero for us
# No SPAN in this

    preds = [mod.pred(rff.transform(norm_dta2[:, batch_s - 1].reshape(lags, 1)))]
    #classic predictions array
    mse = [(preds[0] - endog[0, batch_s]) ** 2] #computing the mse between the predictions and observations - only the first prediction though???

# assumption here is that it only generates a single prediction to this point - have tested this

    for i in range(1, n):
        u = rff.transform(norm_dta2[:, batch_s + i - 1].reshape(lags, 1))  # reshape feature vector ##why does it need to be reshaped???
        # u  holds the features
        d = endog[0, batch_s + i - 1] #d holds the results of the data
        mod.update(u, d)
        preds.append(mod.pred(rff.transform(norm_dta2[:, batch_s + i - 1].reshape(lags, 1))))
        mse.append((preds[i] - endog[0, batch_s + i - 1]) ** 2)

    return preds, mse


print('EWRLS function created')
# %%
#perform EWRLS now
#
strt = time.time()
#
all_mse_EWRLS = []
# D = lags
# span = 300
# window = 50 #this used to be 100000 when we had loads of datapoints
# batch = 1000 #no idea what rhe batch is for
# #
#
#norm_dta2, endog, D, sigma, n, ff, l, roll_size, batch_s, span):
for i in range(2, rounds):
    preds, mse = EWRLS(norm_dta2, # x
                       endog, # y
                       D + 2 ** i,  #D
                       1, #sigma
                       20, #n --- SIZE OF THE WINDOW, this is why preds is 20 long - why would we make 20 predictions?
                       1, #ff
                       0, #l - regularisation term - NOT USED
                      window, #roll_size
                       batch, #batch_s
                       span) #
#this output gives the predictions and the mse
    all_mse_EWRLS.append(np.mean(mse))
    print('RFF EWRLS complete for round ', i)

#print(all_mse_EWRLS)
print(' ALL RFF EWRLS complete')


# %%
x = [D + 2 ** i for i in range(4, rounds)] #this creates the x coordinates
fig = go.Figure()
x_new = np.log(x)
y_new = np.log(all_mse_EWRLS[4:])
#y_new = y_new[5:]
#x_new = x_new[4:]
fig.add_trace(go.Scatter(x=x_new, y=y_new, name="MSE EWRLS"))
fig.update_layout(title="US Inflation QR EWRLS", yaxis_title="MSE EWRLS", xaxis_title="Number of RFF")
#
y_name = all_mse_EWRLS[4:]
x_name = [str(D + 2 ** i) for i in range(4, rounds)]
y_name = [str(round(i, 3)) for i in y_name]
#
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
pio.write_image(fig, 'QR_DD_EWRLS_inflation.png')
#
#
elapsed = time.time() - strt
print(elapsed)

# %%
#here I want to plot my predictions for just one run

fig = plt.figure()

# preferred method for creating 3d axis
ax = fig.add_subplot(111)
ax.plot(preds)
ax.plot(endog)
#plt.show()
