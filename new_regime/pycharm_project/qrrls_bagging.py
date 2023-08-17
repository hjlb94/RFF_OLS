from sampling import sampling
from qrrls_class import QrRlS
from gaussian_rff import GaussianRFF
import numpy as np

def QRRLS_bagging(X, Y, D, sigma, n, ff, l, roll_size, batch_s, n_bags, feature_num, overlap=False):
    # defining an QRRLS function

    # batch_s = 30
    lags = X.shape[0]  # length of the dataset
    all_bags_preds = []
    all_bags_mse = []
    all_bags_mean_mse = []
    models = []

    # first step is to RFF the features
    rff = GaussianRFF(lags, D, sigma)

    # initialise everything by making a single prediction
    X_trans = rff.transform(X[:, :batch_s].reshape(lags, batch_s))  # needs to be batch + n window size
    X_trans = X_trans.T

    features_array, data_array, bags, targets, X_trans = sampling(X_trans, Y, n_bags, feature_num,
                                                                  roll_size, n, overlap)
    # print(len(bags))
    for p in range(0, n_bags):
        mod_QRRLS = QrRlS(bags[p].T[:, :batch_s - 1],  # I think no reshape needed here....
                          targets[p].T[:batch_s - 1],  # y - this is the observations dataset
                          roll_size,  # max_obs - normal variable
                          ff,  # forgetting factor
                          l)
        models.append(mod_QRRLS)

        preds_QRRLS = [mod_QRRLS.pred(bags[p][batch_s - 1].reshape(feature_num, 1))]

        # classic predictions array
        mse_QRRLS = [(preds_QRRLS[0] - targets[p][batch_s - 1]) ** 2]
        all_bags_mse.append(mse_QRRLS)

    # now we need to loop through the time step...

    # time step loop has to be the outermost loop
    for i in range(1, n):

        X_trans = rff.transform(X[:, :batch_s + i].reshape(lags, batch_s + i))
        X_trans = X_trans.T

        features_array, data_array, bags, targets, X_trans = sampling(X_trans, Y, n_bags, feature_num,
                                                                      roll_size, n, overlap)
        # print(len(bags))
        for p in range(0, n_bags):
            u = bags[p][batch_s + i - 1].reshape(feature_num, 1)  # reshape feature vector
            d = targets[p][batch_s + i - 1]  # d holds the results of the data
            models[p].update(u, d)  # update the model

            preds_QRRLS = [models[p].pred(bags[p][batch_s + i - 1].reshape(feature_num, 1))]  # model makes prediction

            # classic predictions array
            mse_QRRLS = [(preds_QRRLS[0] - targets[p][batch_s + i - 1]) ** 2]

            all_bags_mse[p].append(mse_QRRLS)

    for i in range(0, n_bags):
        all_bags_mean_mse.append(np.mean(all_bags_mse[i]))

    return all_bags_mean_mse
