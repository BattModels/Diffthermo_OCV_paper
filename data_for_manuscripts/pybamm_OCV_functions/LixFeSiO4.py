import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LixFeSiO4_OCP(sto):
    """
    diffthermo fit for LixFeSiO4
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    G0 = -342942.5938    
    Omega0 = -140599.2344 
    Omega1 = -133856.3594 
    Omega2 = -89118.7344 
    Omega3 = -233569.4844 
    Omega4 = 19398.2109 
    Omega5 = 279930.0000 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]

    # phase boudary 0
    x_alpha_0 = 0.5397300720214844 
    x_beta_0 = 0.9731568098068237
    mu_coex_0 = -270524.25000 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap_0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap_0 * mu_outside + (1-is_outside_miscibility_gap_0) * mu_coex_0
    return -mu_e/96485.0


