import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def NCO_OCP(sto):
    """
    diffthermo fit for NCO, falling back to regular RK fit
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -38285.5117 
    Omega1 = -14086.7559 
    Omega2 = -20213.9570 
    Omega3 = 22284.7773 
    Omega4 = 23537.8965 
    Omega5 = -9745.8066
    G0 = -389167.1562 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + 1.0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = 1.0 * mu_outside 
    return -mu_e/96485.0


