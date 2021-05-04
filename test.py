# %%
import os
%pylab inline

# %%
## imports
import numpy as np
import JandR_test as jrm
from ODE import rk4
from numba import jit, njit

# %%

t0 = 0
tf= 5 #s
h = 0.001
t = np.linspace(t0,tf,tf/h) # de 0 à 2s avec pas de 1ms
Y0 = np.zeros((6,) )    # pas de CI => 0
# Y0 = np.random.normal(0,1,6)  # CI random

Yrk4 = rk4(jrm.nmm_J_R, t,h, Y0)

%time rk4(jrm.nmm_J_R, t,h, Y0)
# %%
## test njit
rk4_jit = jit()(rk4)
jrnmm_njit = jit()(jrm.nmm_J_R)


# %%
