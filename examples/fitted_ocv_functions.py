import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -11656.055664 # G0 is the pure substance gibbs free energy 
    Omega0 = -2773.382080 
    Omega1 = -5209.178711 
    Omega2 = 5137.673340 
    Omega3 = 5635.966797 
    Omega4 = -20530.710938 
    Omega5 = -18074.468750 
    Omegas =[Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    # phase boundary 0
    x_alpha_0 = 0.5598568916320801
    x_beta_0 = 0.8407307863235474
    mu_coex_0 = -7812.0507812500000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boundary 1
    x_alpha_1 = 0.1563388556241989
    x_beta_1 = 0.3615549206733704
    mu_coex_1 = -12597.9501953125000000
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0  + (1-is_outside_miscibility_gap_1)*mu_coex_1 )
    return -mu_e/96485.0



