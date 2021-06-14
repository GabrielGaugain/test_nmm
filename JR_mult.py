import numpy as np
import matplotlib.pyplot as plt
import random
from ODE import rk4
import timeit
from loguru import logger



def sigmo(mu, e_0=2.5, r=0.56, mu0=6 ):

    return 2*e_0/(1+ np.exp(-r*(mu - mu0)) ) 


def mult_JRM(Y0, t, K,  A=3.25, a =100, B=22, b=50, ad=33, C=135, delayed=1):

    assert isinstance(K, np.ndarray), " K must be a NxN array containing connectivity values between populations "
    assert (K.ndim ==2)&(K.shape[0]==K.shape[1]), " K must be a [ N x N ] array : 2D matrix and same number of lines and columns !!! "
    N = K.shape[0] # N_pop
    
    if Y0.ndim >1 :
        assert (Y0.shape[1] ==N )&(Y0.shape[0] ==8 ), "Y0 and dY0 must be a 1D array of 4*N_pop values or a 2D [4 x N] array"
        # Y = Y.flatten()
        Y = Y0
    
    else :
        assert (Y0.size == 8*N), "Y0 must have 4 * N_pop values if it is un column"
        Y = Y0.reshape((8,N))
        #dY0.reshape((4,N))

    M= np.size(Y)  # N*8
    #Y =  np.zeros((np.size(Y0), np.size(t)))
    dY_dt =  np.zeros(np.shape(Y))

    ## gaussian white noise for the N pop 
    # P = np.random.normal(loc=240, scale=40, size =N)
    C = np.array([1, 0.8, 0.25, 0.25])*C

    if delayed:
        stim = np.random.normal(loc=240, scale=40) + K @ Y[6,:] # must be 0 on diag 
    else : 
        stim = np.random.normal(loc=240, scale=40) + K @ sigmo(Y[1,:]-Y[2,:]) # must be 0 on diag 
        
    dY_dt[0:3,:] = Y[3:6,:]    
    # dY_dt[1,:] = Y[4,:]
    # dY_dt[2,:] = Y[5,:]
    
    dY_dt[3,:] = A*a*sigmo(Y[1,:]- Y[2,:]) - 2*a*Y[3,:] - a**2 * Y[0,:]    

    dY_dt[4,:] = A*a*(C[1]*sigmo(C[0]*Y[0,:]) + stim ) - 2*a*Y[4,:] - a**2 * Y[1,:]   \
                    # + A*a*(P + K@sigmo(Y[1,:] - Y[2,:]))
                    # + A*a*( P + K @ Y[6,:] )
    
    dY_dt[5,:] = B*b*C[3]*sigmo(C[2]*Y[0,:]) - 2*b*Y[5,:] - b**2 * Y[2,:]   

    ## equation for output to other pop
    dY_dt[6,:] = Y[7,:]

    dY_dt[7,:] = A*ad*sigmo( Y[1,:] - Y[2,:] ) - 2*ad*Y[7,:] - ad**2 * Y[6,:]

    if Y0.ndim ==1:
        return dY_dt.flatten()


    return dY_dt
    ## END mult_JRM()


@logger.catch
def main():

    t0 = 0
    tf= 5 #s
    h = 0.001
    t = np.arange(t0,tf,h) # de 0 à 2s avec pas de 1ms
    N = 2
    Y0 = np.zeros((8,N) )    # pas de CI => 0
    # K= 25*(np.ones((N,N)) - np.eye(N))
    K = np.array([[0, 70],[70 , 0]])

    print("K : ",K)
    print("K @ [1 2]", K@np.array([1,2]))

    #dY0 = np.zeros((4,2) )    # pas de CI => 0
    # Y0 = np.random.normal(0,1,6)  # CI random

    Yrk4 = rk4(mult_JRM, t,h, Y0, K=K )
    Yrk4 = Yrk4.reshape((8,K.shape[0],t.size))
    # measure_time(20,rep=2, number=5)

    plot = 1
    if plot ==1:
        plt.figure()
        plt.plot(t, Yrk4[1,0,:]-Yrk4[2,0,:])
        plt.plot(t, Yrk4[1,1,:]-Yrk4[2,1,:])


        plt.figure()
        plt.plot(t,Yrk4[6,0,:])
        plt.show()




    return
    ## END main()



def measure_time(N_pop,rep = 5,number=5):
    Setup_code = '''
from JR_mult import mult_JRM
from ODE import rk4
import numpy as np
    '''

    Test_code = f'''
t0 = 0
tf= 5 #s
h = 0.001
t = np.arange(t0,tf,h) 
N={N_pop}
Y0 = np.zeros((8,N) )    # pas de CI => 0
K= 25*(np.ones((N,N)) - np.eye(N))
Yrk4 = rk4(mult_JRM, t,h, Y0, K=K )
    '''

    times = timeit.repeat(setup=Setup_code,
                            stmt=Test_code,
                            repeat=rep,
                            number=number,
                            )
    print( f'mean time for each calculation with {N_pop} populations : {np.mean(np.array(times)/number)} seconds')
    


if __name__ == "__main__":

    main()



