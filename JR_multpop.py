import numpy as np
import matplotlib.pyplot as plt
import random
from ODE import rk4
import timeit
from loguru import logger



def sigmo(mu, e_0=2.5, r=0.56, mu0=6 ):

    return 2*e_0/(1+ np.exp(-r*(mu - mu0)) ) 


def mult_JRM(Y, t, K,  A=3.25, a =100, B=22, b=50, ad=30, C=135):

    assert isinstance(K, np.ndarray), " K must be a NxN array containing connectivity values between populations "
    assert (K.ndim ==2)&(K.shape[0]==K.shape[1]), " K must be a [ N x N ] array : 2D matrix and same number of lines and columns !!! "
    N = K.shape[0] # N_pop
    
    if Y.ndim >1 :
        assert (Y.shape[1] ==N )&(Y.shape[0] ==8 ), "Y0 and dY0 must be a 1D array of 4*N_pop values or a 2D [4 x N] array"
        Y = Y.flatten()
    
    else :
        assert (Y.size == 8*N), "Y0 must have 4 * N_pop values if it is un column"
        #Y0.reshape((4,N))
        #dY0.reshape((4,N))

    M= np.size(Y)  # N*8
    #Y =  np.zeros((np.size(Y0), np.size(t)))
    dY_dt =  np.zeros((np.size(Y),))

    ## gaussian white noise for the N pop 
    P = np.random.normal(loc=240, scale=40, size =N)
    C = np.array([1, 0.8, 0.25, 0.25])*C

    #stim = np.random.normal(loc=240, scale=40) + K[n,:] @ Y[3,:,] # must be 0 on diag 
    pyr = [n%8==0 for n in range(M)]
    pyr_deriv = [n%8==3 for n in range(M)]

    INE = [n%8==1 for n in range(M)]
    INE_deriv = [n%8==4 for n in range(M)]
    
    INI = [n%8==2 for n in range(M)]
    INI_deriv = [n%8==5 for n in range(M)]

    conn = [n%8==6 for n in range(M)]
    conn_deriv = [n%8==7 for n in range(M)]


    dY_dt[pyr] = Y[pyr_deriv]    
    dY_dt[INE] = Y[INE_deriv]
    dY_dt[INI] = Y[INI_deriv]
    
    dY_dt[pyr_deriv] = A*a*sigmo(Y[INE]- Y[INI]) - 2*a*Y[pyr_deriv] - a**2 * Y[pyr]     
    dY_dt[INE_deriv] = A*a*C[1]*sigmo(C[0]*Y[pyr]) - 2*a*Y[INE_deriv] - a**2 * Y[INE]   \
                     + A*a*( P + K @ Y[conn] )
    dY_dt[INI_deriv] = B*b*C[3]*sigmo(C[2]*Y[pyr]) - 2*b*Y[INI_deriv] - b**2 * Y[INI]   

    ## equation for output to other pop
    dY_dt[conn]       = Y[conn_deriv]
    dY_dt[conn_deriv] = A*ad*sigmo(Y[INE] - Y[INI]) - 2*ad*Y[conn_deriv] - ad**2 * Y[conn]

    return dY_dt
    ## END mult_JRM()

def measure_time(rep = 5,number=5):
    Setup_code = '''
from JR_multpop import mult_JRM
from ODE import rk4
import numpy as np
    '''

    Test_code = '''
t0 = 0
tf= 5 #s
h = 0.001
t = np.arange(t0,tf,h) 
N=70
Y0 = np.zeros((8*N,) )    # pas de CI => 0
K= np.ones((N,N))   
Yrk4 = rk4(mult_JRM, t,h, Y0, K=K )
    '''

    times = timeit.repeat(setup=Setup_code,
                            stmt=Test_code,
                            repeat=rep,
                            number=number,
                            )
    print( times)

@logger.catch
def main():

    t0 = 0
    tf= 5 #s
    h = 0.001
    t = np.arange(t0,tf,h) # de 0 à 2s avec pas de 1ms
    N = 2
    Y0 = np.zeros((8*N,) )    # pas de CI => 0
    K= 60*(np.ones((N,N)) - np.eye(N))
    #K = np.array([[0, 10],[100 , 0]])
    #dY0 = np.zeros((4,2) )    # pas de CI => 0
    # Y0 = np.random.normal(0,1,6)  # CI random

    Yrk4 = rk4(mult_JRM, t,h, Y0, K=K )
    #mult_JRM(Y0,dY0,t,np.ones((2,2)))
    #Y, dY = ODE2_rk4(f, g, t=t, h=h, Y0= Y0[0:3], dY0=Y0[3:6])
    
    #measure_time(rep=1, number=1)

    plt.figure()
    plt.plot(t, Yrk4[1,:]-Yrk4[2,:])
    plt.plot(t, Yrk4[9,:]-Yrk4[10,:])
    plt.show()




    return
    ## END main()


if __name__ == "__main__":

    main()