import numpy as np
from scipy.linalg import pinv
from copy import deepcopy

def rls_weights_pinv_sherman_morrison(self, e, x, y, u):

    """
    RLS weight update algorithm with Sherman Morrison\
    
    w_1 : Weight vector at time t-1, w(t-1)
    k : Kalman Gain
    e : error at time t-1, e(t-1)
    A : XX.T
    x : feature vector at time t, x(t)
    u = (self.I - self.A @ self.P) @ x -- what is this for????
    y : target variable at time t, y(t)
    XY : Feature Matrix at time t, X(t)Y(t)
    beta : 
    P : Inverse Matrix
    """

    # using k * beta instead of pinv(A) * c for performance reason

    P = self.P
    w_1 = self.w
    beta = self.beta
    k = self.k
    XY = self.XY
    k_inv = pinv(k)
    k_inv_T = k_inv.T
    u_inv = pinv(u)
    u_inv_T = u_inv.T

    if np.allclose(u, 0):  # checks if the two arguments are the same to a tolerance
        # essentially saying if u doesn't exist
        # normal sherman morrison update/downdate
        if not np.allclose(beta,0):
            return w_1 + 1/beta*(k * e)
        
        # pseudo inverse sherman morrison downdate
        else:
            w_t = (self.I - k @ k_inv - P @ k_inv_T @ x.T + (k_inv @ P @ k_inv_T) * k @ x.T) @ (w_1 + k * y)
            return w_t

    # pseudo inverse sherman morrison update
    else:
        w_t = (self.I - u_inv_T @ x.T) @ (w_1 + k * y) + (beta * u_inv_T - k) @ (u_inv @ XY)
        return w_t
# must remember that this only fires once
# and is called in a loop to make it recursive
    
