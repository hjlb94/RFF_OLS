from copy import deepcopy
from full_givens import Givens_Rotation
from tkinter import TRUE
import numpy as np
from scipy.linalg import pinv
from pi_update import inv_sherman_morrison_simplified
from weight_update import rls_weights_pinv_sherman_morrison
from numpy.linalg import _umath_linalg
from last_row_givens import givens
from givens_rotation_downdate import givens_rot_down


class QrRlS:

    def __init__(self,x,y,max_obs,ff,l):

        """
        x - Initial input Dataset
        y - Initial output Dataset
        max_obs = Rolling window size
        ff = Forgetting factor
        l = lambda(regularization term)
        """

        self.X = x
        self.y = y
        self.dim = len(x)
        self.I = np.eye(self.dim)
        ff = np.sqrt(ff)
        self.ff = ff
        self.l = l
        self.b = 1
        
        # Forgetting factor matrix
        B = np.diag([ff**i for i in range(x.shape[1]-1,-1,-1)])
        self.X = self.X @ B
        self.n_batch = x.shape[1]
        self.Q, self.R = Givens_Rotation(self.X.T)
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ y
        self.z = self.Q.T @ y

        # A and P were used as R and R inverse
        self.A = self.R
        self.P = self.R_inv
        self.max_obs = max_obs
        self.all_Q = deepcopy(self.Q)
        self.i = 1
        
    def update(self,x,y):

        self.X = np.c_[self.X, x]
        self.y = np.r_[self.y,y]
        nobs = np.shape(self.X)[1]
        self.P = (1/self.ff)*self.P
        self.A = self.ff * self.A
        d = x.T @ self.P
        c = x.T @ (np.eye(self.A.shape[1]) - self.P @ self.A)

        # Update for new regime
        if not np.allclose(0,c):
            c_inv = pinv(c)
            self.P = np.c_[self.P - c_inv @ d, c_inv]

        # Update for old regime
        else:
            b_k = 1/(1 + d @ d.T) * self.P @ d.T
            self.P = np.c_[self.P - b_k @ d, b_k]

        self.A = np.r_[self.A, x.T]
        self.Q,self.A = givens(self)
        y = np.array(self.y).reshape(self.X.shape[1],1)
        self.w = self.P @ y
        self.P = self.P @ self.Q.T
        self.i += 1
        
        if nobs > self.max_obs:
            x = self.X[:,0].reshape(self.dim,1)
            self.delete(x,self.y[0])


    def delete(self,x,y):
        """
        x - features which will get deleted (dim x 1)
        y - target which will get deleted (scalar)
        """

        temp = np.allclose(np.eye(self.A.shape[1]),self.A.T @ self.P.T)
        self.X = self.X[:,1:]
        self.Q = givens(self, False)
        self.P = self.P @ self.Q
        self.A = self.Q.T @ self.A
        x = self.A[0,:].reshape(self.dim,1)
        c = np.zeros((self.P.shape[1],1))
        c[0,0] = 1
        k = self.P @ c
        h = x.T @ self.P
        je = x.T @ self.P @ c

        # Deletion for new regime
        if not temp:
            self.P = self.P - k @ pinv(k) @ self.P - self.P @ pinv(h) @ h + (pinv(k) @ self.P @ pinv(h)) * k @ h

        # Deletion for old regime
        else:
            x = -x
            h = x.T @ self.P
            u = (np.eye(self.P.shape[1]) - self.A @ self.P) @ c
            k = self.P @ c
            h_mag = h @ h.T
            u_mag = u.T @ u
            S = (1 + x.T @ self.P @ c)
            p_2 = - ((u_mag)/S * self.P @ h.T) - k
            q_2 = - ((h_mag)/S * u.T - h)
            sigma_2 = h_mag * u_mag + S**2
            self.P = self.P + 1/S * self.P @ h.T @ u.T - S/sigma_2 * p_2 @ q_2
        
        self.P = self.P[:,1:]
        self.A = self.A[1:,:]
        y = np.array(self.y).reshape(self.X.shape[1]+1,1)
        y = self.all_Q.T @ y
        y = y[1:]
        self.w = self.P @ y
        self.all_Q = self.all_Q[1:,1:]
        self.y = self.y[1:]


    def pred(self,x):
        """
        x - features (dim x 1)
        """
        
        if x.shape[1] == 1:
            pred = (x.T @ self.w).item()
            
        else:
            pred = x.T @ self.w
        return pred

    