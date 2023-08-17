##ok so have endog and norm_dta2 in df and not transposed, features in columns - easier to imagine
# now to produce a bagging regime
# first step is to get the features we need for each regime....

import numpy as np
import random

def sampling(X, Y, n_bags, feature_sample_number, roll_size, n, overlap=False):
    features_array = []  # array to hold feature indices
    data_array = []  # array to hold data indices
    bags = []  # holds the bags
    targets = []  # holds the Y values
    lags = X.shape[1]  # length of the dataset
    length = X.shape[0]  # columns of the dataset
    #test_feat = np.arange(0, D)  # total features
    used_feat = []  # used feature array

    feature_list = range(0, X.shape[1])  # list of features for checking

    if not overlap:

        # need to generate the bags from our RFF features
        for i in range(0, n_bags):
            # feature_sampling
            features = np.sort(random.sample(range(0, X.shape[1]), feature_sample_number))
            features_array.append(features)

            # data sampling - removed as unnecessary
            #start_index = random.choice(range(0, (X.shape[0] - roll_size + n)))
            #end_index = start_index + roll_size + n  # data only has to be rolling_window + n predictions in length

            # target_sampling
            targets.append(Y[0][:])

            # [columns][rows]..... .iloc[rows, columns]
            bags.append(np.array(X[:, features]))

        # check
        unused_feat = [n for n in feature_list if n not in np.concatenate(features_array, 0)]

        # if len(unused_feat) == 0:
        #    print('all features used')
        # else:
        #    print('use more bags or increase feature number')

    if overlap:

        # first need to generate the required number of features for overlap
        overlap_features = np.sort(random.sample(range(0, X.shape[1]), int(overlap * feature_sample_number)))

        feature_list = [i for i in feature_list if i not in overlap_features]

        for i in range(0, n_bags):
            # feature_sampling
            features = np.sort(
                random.sample(range(0, X.shape[1]), feature_sample_number - int(overlap * feature_sample_number)))
            features_array.append(features)
            features_array[i] = np.concatenate((features_array[i], overlap_features))

            # data sampling - removed as unnecessary
            # start_index = random.choice(range(0, X.shape[0] - (roll_size + n)))
            # end_index = start_index + roll_size + n #data only has to be rolling_window + n predictions in length

            # target_sampling
            targets.append(Y[0][:])
            # [columns][rows]..... .iloc[rows, columns]
            bags.append(np.array(X[:, features_array[i]]))

        # check
        feature_list = range(0, X.shape[1])  # need original features list
        unused_feat = [n for n in feature_list if n not in np.concatenate(features_array, 0)]

        # if len(unused_feat) == 0:
        #    print('all features used')
        # else:
        #    print('use more bags or increase feature number')

    return features_array, data_array, bags, targets, X




