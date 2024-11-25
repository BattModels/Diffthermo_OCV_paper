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
    G0 = -381555.5625 
    Omega0 = -42432.0820 
    Omega1 = 6579.1157 
    Omega2 = 23361.8047 
    Omega3 = 31590.4922 
    Omega4 = 28378.8008 
    Omega5 = 29163.8047 
    Omega6 = 8955.4033 
    Omega7 = -7601.7798 
    Omega8 = -3563.2253 
    Omega9 = 19743.1211 
    Omega10 = -24798.2754 
    Omega11 = 2853.8174 
    Omega12 = 293.8150 
    Omega13 = -15989.4336 
    Omega14 = 33684.6250 
    Omega15 = -27924.1973 
    Omega16 = 42068.6250 
    Omega17 = -28561.3672 
    Omega18 = -243.7913
    Omega19 = 42760.6289 
    Omega20 = -119639.9375 
    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, Omega9, Omega10, Omega11, Omega12, Omega13, Omega14, Omega15, Omega16, Omega17, Omega18, Omega19, Omega20]
    
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + 1.0 * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = 1.0 * mu_outside 
    return -mu_e/96485.0


