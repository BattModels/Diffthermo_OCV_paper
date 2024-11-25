import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def NCA_OCP(sto):
    """
    diffthermo fit for NCA, falling back to regular RK fit
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -40536.5078 
    Omega1 = 7386.1909 
    Omega2 = 10436.7803 
    Omega3 = 39447.5586 
    Omega4 = 159940.9844 
    Omega5 = 149692.1406 
    G0 = -381002.9688 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + 1.0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = 1.0 * mu_outside 
    return -mu_e/96485.0


