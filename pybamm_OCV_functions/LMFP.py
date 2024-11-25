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
    Omega0 = -145909.7344 
    Omega1 = 74384.1953 
    Omega2 = -110217.6562 
    Omega3 = 283424.2500 
    Omega4 = 4096.3789 
    Omega5 = -488764.0000 
    Omega6 = -118211.7422 
    Omega7 = 876900.6875 
    Omega8 = -463140.0000 
    G0 = -296098.3438   
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8]

    # phase boudary 0
    x_alpha_0 = 0.5106148719787598  
    x_beta_0 = 0.6389647722244263
    mu_coex_0 = -332565.03125 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.7089425325393677  
    x_beta_1 = 0.8747773170471191
    mu_coex_1 = -323988.12500  # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0

