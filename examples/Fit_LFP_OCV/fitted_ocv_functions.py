import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -336660.593750 # G0 is the pure substance gibbs free energy 
    Omega0 = 128051.375000 
    Omega1 = 61734.765625 
    Omega2 = -290618.718750 
    Omega3 = -163956.625000 
    Omegas =[Omega0, Omega1, Omega2, Omega3]
    # phase boundary 0
    x_alpha_0 = 0.0811861157417297
    x_beta_0 = 0.9259547591209412
    mu_coex_0 = -328734.6875000000000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0 )
    return -mu_e/96485.0



