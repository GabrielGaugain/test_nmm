import numpy as np
import matplotlib.pyplot as plt
import random
from ODE import rk4

C=135




def sigmo(mu, e_0=2.5, r=0.56, mu0=6 ):

    return 2*e_0/(1+ np.exp(-r*(mu - mu0)) ) 



def nmm_J_R(Y,t, A=3.25, a =100 , B=22 ,b= 50, C=135):
    """
    Y = (y0     )
        (y1     )
        (y2     )
        (dy0_dt )
        (dy1_dt )
        (dy2_dt )
    
    """

    C = np.array([1, 0.8, 0.25, 0.25])*C

    P = np.random.normal(loc=240, scale=40)
    #P = random.gauss(240,40)
    dY_dt  = np.zeros((np.size(Y),))

    dY_dt[0] = Y[3]    
    dY_dt[1] = Y[4]
    dY_dt[2] = Y[5]

    dY_dt[3] = A*a*sigmo(Y[1]- Y[2]) - 2*a*Y[3] - a**2 * Y[0]     
    dY_dt[4] = A*a*(P + C[1]*sigmo(C[0]*Y[0])) - 2*a*Y[4] - a**2 * Y[1]     
    dY_dt[5] = B*b*C[3]*sigmo(C[2]*Y[0]) - 2*b*Y[5] - b**2 * Y[2]     

    return dY_dt



def main():

    t0 = 0
    tf= 5 #s
    h = 0.001
    t = np.linspace(t0,tf,tf/h) # de 0 à 2s avec pas de 1ms
    Y0 = np.zeros((6,) )    # pas de CI => 0
    # Y0 = np.random.normal(0,1,6)  # CI random

    Yrk4 = rk4(nmm_J_R, t,h, Y0)

    #Y, dY = ODE2_rk4(f, g, t=t, h=h, Y0= Y0[0:3], dY0=Y0[3:6])

    plt.figure()
    plt.plot(t, Yrk4[1,:]-Yrk4[2,:])


    Y02 = np.zeros((8,))
    Yrk42 = rk4(nmm_J_R_JULIEN,t,h,Y02)

    plt.figure()
    plt.plot(t, Yrk42[1,:]-Yrk42[2,:])

    plt.show()




    return



if __name__ == "__main__":

    main()