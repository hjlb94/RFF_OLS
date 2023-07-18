import numpy as np

def RFF(X, D, sameFunction):
    np.random.seed(0)
    n_features = len(X)
    
    if sameFunction:
        omegas1 = (np.random.normal(size=(1, D)))
        omegas = omegas1
        
        for i in range(n_features-1):
            omegas = np.vstack([omegas, omegas1])
            
    else:
        omegas = (np.random.normal(size=(n_features, D)))
        
    b = np.random.uniform(0, 2 * np.pi, size=(D,1))
    X_new = np.transpose(omegas)@X+b
    X_new = np.cos(X_new)
    X_new *= np.sqrt(2) / np.sqrt(D)
    
    return X_new
