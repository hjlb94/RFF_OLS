import kaleido
import pandas as pd
import numpy as np
from ewrls_rolling import EwRls
#import matplotlib.pyplot as plt
from gau_rff import GaussianRFF
#from scipy.stats import pearsonr
import time
import plotly.graph_objects as go
import plotly.io as pio

print('imports done ')


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
print('data loading and normalisation complete')

## define the EWRLs
D = 1000 #row length of the RFF features output
lags = norm_dta2.shape[0] #this is a problem piece
span = 300
window = 50
batch = 10 #this is the column number of the RFF feature output
sigma = 1

# %%
# here I will test the rff function

# definition    EWRLS(norm_dta2, endog, D,          sigma,  n, ff, l, roll_size, batch_s, span):
# use           EWRLS(norm_exog, endog, D + 2 ** i, 1,      20, 1, 0, window,    batch,   span)


rff = GaussianRFF(lags, D, sigma) #creating a method called RFF
#works absolutely fine up to here

#the input of the features matrix is
rff_features = rff.transform(norm_dta2[:, :batch].reshape(lags, batch))

print('RFF features matrix created')
