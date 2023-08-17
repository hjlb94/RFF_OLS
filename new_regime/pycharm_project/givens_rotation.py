## Givens Rotation ##

import numpy as np

def Givens_Rotation(R):
    n = len(R)
    Q = np.identity(n)
    for j in range(R.shape[1]):
        for i in range(j + 1, n):
            x = R[j][j]
            y = R[i][j]
            r = np.sqrt(x ** 2 + y ** 2)
            if r != 0:
                c = x / r
                s = -y / r
                I = np.identity(n)
                I[i, i] = c
                I[j, j] = c
                I[i, j] = s
                I[j, i] = -s
                Q = Q @ I.T
                R = I @ R

    # return R[:n,:]
    return Q, R