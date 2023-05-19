import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LMFP_OCV(sto):
    """
    diffthermo fit for LMFP
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -107495.0625 
    Omega1 = 26662.8164 
    Omega2 = 25543.9824 
    Omega3 = 47758.4688 
    Omega4 = -199459.4688 
    Omega5 = 299251.1875 
    Omega6 = -163829.0781 
    G0 = -315364.2500  
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6]

    # phase boudary 0
    x_alpha_0 = 0.5021629333496094  
    x_beta_0 = 0.8771189451217651
    mu_coex_0 = -328099.15625 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.0451637208461761 
    x_beta_1 = 0.1859853267669678
    mu_coex_1 = -386181.37500  # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0

