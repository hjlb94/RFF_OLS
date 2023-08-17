import numpy as np

def givens_rot_down(R,pinv_R,obs_d):
    q=pinv_R.T@obs_d
    n = len(R)
    k = len(R[0])
    R = np.c_[np.zeros(n).T,R]
    R = np.r_[np.zeros((1,k+1)),R]
    R[1:,0] = q.ravel()
    n = len(R)
    Q = np.identity(n)
    for j in range(0,1):
        for i in np.arange(n-1,0,-1):
            x=R[j][j]
            y=R[i][j]
            r=np.sqrt(x**2+y**2)
            if r!=0:
              c = x/r
              s = -y/r
              I = np.identity(n)
              I[i,i] = c
              I[j,j] = c
              I[i,j] = s
              I[j,i] = -s
              Q = Q@I.T
              R = I@R
    
    return Q,R[1:-1,1:]
