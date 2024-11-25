import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def NMC_OCP(sto):
    """
    diffthermo fit for NMC, falling back to regular RK fit
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -95399.1641 
    Omega1 = -51806.6875
    Omega2 = -29817.8145 
    Omega3 = -50930.4570 
    Omega4 = -90481.8828 
    Omega5 = -61381.0547 
    G0 = -408616.8438 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + 1.0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = 1.0 * mu_outside 
    return -mu_e/96485.0


