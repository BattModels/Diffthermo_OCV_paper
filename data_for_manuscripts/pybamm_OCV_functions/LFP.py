import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LFP_OCP(sto):
    """
    diffthermo fit for LFP
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    G0 = -336668.3750 # G0 is the pure substance gibbs free energy 
    Omega0 = 128205.6641
    Omega1 = 61896.4375
    Omega2 = -290954.8125
    Omega3 = -164259.3750
    Omegas = [Omega0, Omega1, Omega2, Omega3]
    # phase coexistence chemical potential
    mu_coex = -328734.78125
    # phase boudary
    x_alpha = 0.08116829395294189453
    x_beta = 0.92593580484390258789
    is_outside_miscibility_gap = (sto<x_alpha) + (sto>x_beta)
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap * mu_outside + (1-is_outside_miscibility_gap) * mu_coex
    return -mu_e/96485.0


