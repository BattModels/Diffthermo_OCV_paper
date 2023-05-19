import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LTO_OCV(sto):
    """
    diffthermo fit for LTO
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = 14025.5664 
    Omega1 = 15593.5771 
    Omega2 = -100475.7578 
    Omega3 = -26389.4648 
    G0 = -155432.9219 
 
    Omegas = [Omega0, Omega1, Omega2, Omega3]

    # phase boudary 0
    x_alpha_0 = 0.1186100244522095  
    x_beta_0 = 0.8482373952865601
    mu_coex_0 = -155770.73438 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap_0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap_0 * mu_outside + (1-is_outside_miscibility_gap_0) * mu_coex_0
    return -mu_e/96485.0


