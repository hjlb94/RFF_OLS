import pandas as pd
import numpy as np
from qrrls_bagging import QRRLS_bagging
import time

dta2 = pd.read_csv('/Users/hugobarette/Documents/Uni/Summer_project/InflationData/fredmd_transf_df.csv')
dta2_A = dta2["RPI"].to_numpy()
# dta2_A = dta2_A[:100000]
rpi = dta2_A.reshape(1, len(dta2))

# lambda function to calc the returns
ret = lambda x: (x[0, 1:] - x[0, :-1]) / x[0, :-1]

# calculate the return and store in variable
exog = ret(rpi).reshape(1, rpi.shape[
    1] - 1)  # these are the results, not using the same stuff as previous years wrt the lag matrix

# initialise normalised array of correct size
norm_exog = np.zeros((1, exog.shape[1]))  # why does this require two brackets??

# normalise the target variable
for i in range(2, exog.shape[1]):
    norm_exog[0, i - 2] = (exog[0, i - 1] - exog[0, :i].mean()) / exog[0, :i].std()
# have normalised observations now.... the normalisation is rolling
endog = norm_exog

# next we need the features dataset
dta2 = dta2.drop(columns=['RPI', 'date'])  # drop unnecessary columns

# fill na
dta2 = dta2.fillna(0)

# create empty arrays
norm_dta2 = np.zeros((dta2.shape[0], dta2.shape[1] - 1))
norm_dta2 = pd.DataFrame(norm_dta2)

# creates the normalised features matrix
for n in range(0, 125):
    for i in range(2, dta2.shape[0]):
        # iloc is row, column
        norm_dta2[n][i - 2] = (dta2.iloc[i - 1, n] - dta2.iloc[:i, n].mean()) / dta2.iloc[:i, n].std()

# necessary to drop first line
norm_dta2 = norm_dta2[:][1:].fillna(0)

norm_dta2 = np.array(norm_dta2.T)
# first section test
print('data imported and normalised')

# %% DATA IMPORT FINISHED

D = 4096
sigma = 1
n = 20
ff = 1
l = 0
roll_size = 30
batch_s = 30
round_start = 0
rounds = 2
overlap = False
mse_array = []
bag_num = [10, 20, 50, 100, 150, 200]  # , 500]
overlap_array = [0, 0.1, 0.2, 0.3]  # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

X1, Y1 = np.meshgrid(bag_num, overlap_array)

# QRRLS_bagging(X, Y, D, sigma, n, ff, l, roll_size, batch_s, n_bags, feature_num, overlap=False):

for q in overlap_array:
    mse_array_temp = []
    for p in bag_num:
        start = time.time()
        mse = QRRLS_bagging(norm_dta2, endog, D, sigma, n, ff, l, roll_size,
                            batch_s, p, 100, q)
        mse_array_temp.append(np.mean(mse))
        print('overlap ', q, ', bag_num ', p, ', complete in ', time.time() - start)
    mse_array.append(mse_array_temp)
print('')

print('complete')

fig1 = plt.figure(figsize=(15, 6), dpi=200)
ax1 = plt.axes(projection='3d')
# X1, Y1 = np.meshgrid(x_vals, mse_array)
ax1.plot_surface(X1, Y1, np.array(mse_array), rstride=1, cstride=1,
                 cmap='viridis', edgecolor='none')
# plt.plot(x_vals, mse_array, 'o-')
# plt.ylim([0, np.max(mse_array)*1.1])
ax1.set_xlabel('Bag number')
ax1.set_ylabel('Overlap Ratio')
ax1.set_zlabel('MSE')
ax1.set_title('4096 features RFF QR Decomp - 100 features per bag')
# plt.savefig('bags_overlap_3d.jpg', bbox_inches = 'tight')

plt.show()