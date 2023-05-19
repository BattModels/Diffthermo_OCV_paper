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
    G0 = -12254.9414 
    Omega0 = -2368.2908 
    Omega1 = -4401.7715 
    Omega2 = 13879.9395 
    Omega3 = -52430.7227 
    Omega4 = -43530.1719 
    Omega5 = 139934.6719 
    Omega6 = 19018.1602 
    Omega7 = -118288.8906 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7]

    # phase boudary 0
    x_alpha_0 = 0.5450866818428040  
    x_beta_0 = 0.9030561447143555
    mu_coex_0 = -7666.44336 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.2775516510009766  
    x_beta_1 = 0.4738220572471619 
    mu_coex_1 = -10955.86426 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0


