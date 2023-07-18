import numpy as np
from scipy.linalg import pinv
from pi_update import inv_sherman_morrison_simplified
from weight_update import rls_weights_pinv_sherman_morrison

class EwRls:

    def __init__(self,x, y, max_obs, ff, l, span = False):
        """
        dim - x dimension
        X - Initial Dataset
        max_obs = Rolling window size
        ff = Forgetting factor
        l = lambda(regularization term)
        """

        if span != False:
            ff = 1 - 2/(span + 1)

        self.X = x # features matrix - norm_dta2 for me
        self.y = y # observations vector - endog
        self.dim = len(x) # dimension of the matrix
        self.I = np.eye(self.dim) # identity matrix
        B = np.diag([ff**i for i in range(x.shape[1]-1,-1,-1)]) # gives diagonal matrix of the betas (forgetting factor I THINK)
        self.A = x @ B @ x.T + l*self.I
        self.P = pinv(self.A)
        self.ff = ff
        self.l = l
        
        if np.size(y) == 1:
            self.w = self.P @ x * y
            self.XY = x * y
        else:
            y = y.reshape(x.shape[1], 1) # why is this step necessary?? maybe to make the results and the features the same length?
            self.w = self.P @ x @ B @ y
            self.XY = x @ y

        self.max_obs = max_obs


    def update(self,x,y): #when t
        """
        x - features (dim x 1)
        y - target (scalar)
        """
    
        self.X = np.c_[self.X, x] # c_ this adds the two arguments along the second access
        self.y = np.append(self.y,y)
        u = (self.I - self.A @ self.P) @ x
        self.P = (1/self.ff)*self.P
        self.beta = (1 + (x.T @ self.P @ x)).item()
        self.k = self.P @ x
        self.XY = self.XY + x*y
        e = y - x.T @ self.w
        self.w = rls_weights_pinv_sherman_morrison(self, e, x, y, u)
        self.P = inv_sherman_morrison_simplified(self,u)
        self.A = self.A*self.ff + x@x.T
        nobs = np.shape(self.X)[1]


        if nobs > self.max_obs:
            x = self.X[:,0].reshape(self.dim,1)
            self.delete(x,self.y[0]) #ooooh I like this style of coding - referencing a method from the class in another method, very nice!

    def delete(self,x,y):
        """
        x - features which will get deleted (dim x 1)
        y - target which will get deleted (scalar)
        """

        x = x * np.sqrt(self.ff**(self.max_obs))
        u = (self.I - self.A @ self.P) @ x
        self.beta = (1 - (x.T @ self.P @ x)).item()
        self.k = self.P @ x
        e = -y + x.T @ self.w
        self.XY = self.XY - x*y
        self.w = rls_weights_pinv_sherman_morrison(self, e, x, y, u)
        self.P = inv_sherman_morrison_simplified(self , u, False)
        self.A = self.A - x@x.T
        self.X = self.X[:,1:]
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

    