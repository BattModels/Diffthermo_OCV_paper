import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -12250.294922 # G0 is the pure substance gibbs free energy 
    Omega0 = -2376.557373 
    Omega1 = -4394.042969 
    Omega2 = 13664.444336 
    Omega3 = -52024.406250 
    Omega4 = -42901.992188 
    Omega5 = 138761.125000 
    Omega6 = 18488.441406 
    Omega7 = -117369.820312 
    Omegas =[Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7]
    # phase boundary 0
    x_alpha_0 = 0.5454377532005310
    x_beta_0 = 0.9029746651649475
    mu_coex_0 = -7666.7456054687500000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boundary 1
    x_alpha_1 = 0.9098722934722900
    x_beta_1 = 0.8019378781318665
    mu_coex_1 = -11549.5488281250000000
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)
    # phase boundary 2
    x_alpha_2 = 0.2783287167549133
    x_beta_2 = 0.4735463261604309
    mu_coex_2 = -10953.1787109375000000
    is_outside_miscibility_gap_2 = (sto<x_alpha_2) + (sto>x_beta_2)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1 * is_outside_miscibility_gap_2     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0  + (1-is_outside_miscibility_gap_1)*mu_coex_1  + (1-is_outside_miscibility_gap_2)*mu_coex_2 )
    return -mu_e/96485.0



