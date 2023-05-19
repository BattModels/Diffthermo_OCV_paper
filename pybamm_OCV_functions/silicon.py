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
    Omega0 = 10393.8350 
    Omega1 = -43130.8438 
    Omega2 = 1136.0477 
    Omega3 = 155547.7812 
    Omega4 = -20951.0508 
    Omega5 = -141700.1250 
    G0 = -9296.2520  

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]

    # phase boudary 0
    x_alpha_0 = 0.3440414667129517  
    x_beta_0 = 0.9013724327087402
    mu_coex_0 = -7715.14844 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.0473743975162506  
    x_beta_1 = 0.3288094401359558
    mu_coex_1 = -10386.01270 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0


