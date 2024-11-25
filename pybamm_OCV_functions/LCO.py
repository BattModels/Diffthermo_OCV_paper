import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LCO_OCP(sto):
    """
    diffthermo fit for LCO
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -162566.8281 
    Omega1 = -76570.4219 
    Omega2 = -79813.6484 
    Omega3 = -61739.2266 
    Omega4 = -107894.6172 
    Omega5 = -226719.6719 
    Omega6 = 104857.2578 
    Omega7 = 51303.8125 
    Omega8 = -536150.8750 
    G0 = -430036.8750  
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8]

    # phase boudary 0
    x_alpha_0 = 0.6937953829765320  
    x_beta_0 = 0.9380667805671692
    mu_coex_0 = -355379.81250 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap_0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap_0 * mu_outside + (1-is_outside_miscibility_gap_0) * mu_coex_0
    return -mu_e/96485.0


