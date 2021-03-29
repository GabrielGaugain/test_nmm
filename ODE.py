import numpy as np



def rk4(f, t,h, y0):

    N = np.size(t)
    Y = np.zeros((np.size(y0), N))
    Y[:,0] = y0

    for n in range(N-1):
        
        yn = Y[:,n]
        tn = t[n]
        
        k1 = h*f(yn, tn)
        k2 = h*f(yn + 0.5*k1, tn + h/2 )
        k3 = h*f(yn + 0.5*k2, tn + h/2 )
        k4 = h*f(yn + k3  , tn + h   )

        dy = (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        Y[:,n+1] = yn + dy
    return Y
