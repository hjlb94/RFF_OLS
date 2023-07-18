import numpy as np
from copy import deepcopy

def givens(self,update=True):
    if update:
        A = self.A
        if A.shape[0] > A.shape[1]:
            diag = A.shape[1]-1

        else:
            diag = A.shape[0]-1

        G = np.identity(A.shape[0])
        all_Q = deepcopy(self.all_Q)
        all_Q = np.concatenate((all_Q,np.zeros((all_Q.shape[0],1))),axis=1)
        all_Q = np.concatenate((all_Q,np.zeros((1,all_Q.shape[1]))),axis=0)
        #all_Q = np.concatenate((all_Q,np.zeros((len(A)-1,1))),axis=1)
        #all_Q = np.concatenate((all_Q,np.zeros((1,len(A)))),axis=0)
        all_Q[-1,-1] = 1
        Q = deepcopy(G)

        for i in range(diag):
            x = A[i,i]
            y = A[-1,i]
            r = np.sqrt(x**2 + y**2)
            c = x / r
            s = -y / r
            G[i,i] = c
            G[-1,-1] = c
            G[i,-1] = -s
            G[-1,i] = s
            A = G @ A
            Q = Q @ G.T
            G = np.identity(A.shape[0])

        #A = np.round_(A,decimals = 10)
        self.all_Q = all_Q @ Q
        return Q.T,A

    else:

        P = self.P
        G = np.identity(P.shape[1])
        G_all = deepcopy(G)
        diag = P.shape[1]-1
        A = self.A
        q = self.all_Q[0,:].reshape(self.all_Q.shape[1],1)

        for i in range(diag,0,-1):
            x = q[0,0]
            y = q[i,0]
            r = np.sqrt(x**2 + y**2)
            c = x / r
            s = -y / r
            G[i,i] = c
            G[0,0] = c
            G[0,i] = -s
            G[i,0] = s
            A = G @ A
            G_all = G_all @ G.T
            q = G @ q
            G = np.identity(self.all_Q.shape[0])

        self.all_Q = self.all_Q @ G_all

        #A = np.round_(A,decimals = 10)
        return G_all

       