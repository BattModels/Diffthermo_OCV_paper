import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -11686.997070 # G0 is the pure substance gibbs free energy 
    Omega0 = -2697.453369 
    Omega1 = -5728.828125 
    Omega2 = 4744.390137 
    Omega3 = 7466.139160 
    Omega4 = -19736.783203 
    Omega5 = -20078.644531 
    Omegas =[Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    # phase boundary 0
    x_alpha_0 = 0.1569776684045792
    x_beta_0 = 0.3498985171318054
    mu_coex_0 = -12697.3496093750000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boundary 1
    x_alpha_1 = 0.5453399419784546
    x_beta_1 = 0.8350558280944824
    mu_coex_1 = -7888.4428710937500000
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0  + (1-is_outside_miscibility_gap_1)*mu_coex_1 )
    return -mu_e/96485.0



