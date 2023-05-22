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
    Omega0 = 10925.7754 
    Omega1 = -10595.5332 
    Omega2 = -82399.9219 
    Omega3 = 49068.6367 
    Omega4 = 207297.7188 
    Omega5  = -52962.5977 
    Omega6  = -163353.1875 
    G0 = -8911.9854  

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6]

    # phase boudary 0
    x_alpha_0 = 0.2721128463745117  
    x_beta_0 = 0.7885768413543701
    mu_coex_0 = -8434.79102 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.0457834899425507  
    x_beta_1 = 0.2572654187679291
    mu_coex_1 = -10762.76562 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)
    # phase boudary 2
    x_alpha_2 = 0.8234204649925232  
    x_beta_2 = 0.9572074413299561
    mu_coex_2 = -4658.05078 # phase coexistence chemical potential
    is_outside_miscibility_gap_2 = (sto<x_alpha_2) + (sto>x_beta_2)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1 * is_outside_miscibility_gap_2

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1 + (1-is_outside_miscibility_gap_2)*mu_coex_2)
    return -mu_e/96485.0


