import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LMO_OCP(sto):
    """
    diffthermo fit for LMO, falling back to regular RK fit
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -45261.5703 
    Omega1 = -24621.5312 
    Omega2 = -31228.3926 
    Omega3 = -47535.9023 
    Omega4 = -35774.7656 
    G0 = -398802.0312 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4]
    
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + 1.0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = 1.0 * mu_outside 
    return -mu_e/96485.0


