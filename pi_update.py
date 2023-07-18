from scipy.linalg import pinv
import numpy as np
from copy import deepcopy

def inv_sherman_morrison_simplified(self, u, update = True):

    beta = self.beta
    pinv_A = self.P
    k = deepcopy(self.k)
    beta = self.beta
    pinv_A = self.P
    k_T = k.T
    pinv_k = pinv(k)
    pinv_k_T = pinv_k.T
    pinv_u = pinv(u)

    if (np.allclose(u,0)):

        if update:
            return (pinv_A - (1/beta)*k@k_T)
            
        elif not np.allclose(beta,0):
            return (pinv_A + (1/beta)*k@k_T)

        else:
            return (pinv_A - (k @ pinv_k @ pinv_A) - pinv_A@pinv_k_T@k_T + (pinv_k @ pinv_A @ pinv_k_T) * k @ k_T)

    else:
        return (pinv_A - k@pinv_u - pinv_u.T@k_T + (beta)*pinv_u.T@pinv_u)
    