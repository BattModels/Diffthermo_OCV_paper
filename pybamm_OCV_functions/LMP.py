import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LMP_OCV(sto):
    """
    diffthermo fit for LMP
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -70752.6250 
    Omega1 = 67894.5547 
    Omega2 = -41201.6484 
    Omega3 = 22804.5215 
    Omega4 = -35592.1719 
    G0 = -353011.1250 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4]

    # phase boudary 0
    x_alpha_0 = 0.1076388657093048  
    x_beta_0 = 0.4588075876235962
    mu_coex_0 = -389349.84375 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap_0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap_0 * mu_outside + (1-is_outside_miscibility_gap_0) * mu_coex_0
    return -mu_e/96485.0


