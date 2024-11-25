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
    G0 = -9086.5371    
    Omega0 = 1006.7307 
    Omega1 = -869.3273 
    Omega2 = -2671.1729 
    Omega3 = -2905.4810 
    Omega4 = 4745.7280 
    Omega5 = 8474.4834 
    Omega6 = -4934.8208 
    Omega7 = -13251.8213 
    Omega8 = -14416.9688 
    Omega9 = -13249.3545 
    Omega10 = -7897.8389 
    Omega11 = -108.8177 
    Omega12 = 11091.6396 
    Omega13 = 22201.2773 
    Omega14 = 15541.2148 
    Omega15 = 8058.5503 
    Omega16 = 17959.3809 
    Omega17 = 6066.4590 
    Omega18 = -5868.1895 
    Omega19 = -41169.6758 
    Omega20 = -44457.4414

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, Omega9, Omega10, Omega11, Omega12, Omega13, Omega14, Omega15, Omega16, Omega17, Omega18, Omega19, Omega20]
    

    # phase boudary 0
    x_alpha_0 = 0.1380511522293091  
    x_beta_0 = 0.2629371881484985
    mu_coex_0 = -10175.51367 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 )
    return -mu_e/96485.0


