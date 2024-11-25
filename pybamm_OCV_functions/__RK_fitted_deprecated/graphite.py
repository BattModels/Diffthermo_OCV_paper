 import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def graphite_OCP(sto):
    """
    diffthermo fit for graphite
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -3809.5845 
    Omega1 = -4032.4138 
    Omega2 = 6000.8306 
    Omega3 = -11625.2646 
    Omega4 = -62671.4648 
    Omega5 = 25442.2031 
    Omega6 = 116366.1172 
    Omega7 = -26409.8652 
    Omega8 = -90384.7344 
    G0 = -11595.3965 

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8]

    # phase boudary 0
    x_alpha_0 = 0.5630063414573669  
    x_beta_0 = 0.8214209079742432
    mu_coex_0 = -7810.95215 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.2831532061100006  
    x_beta_1 = 0.4294382333755493
    mu_coex_1 = -11124.01855 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0


