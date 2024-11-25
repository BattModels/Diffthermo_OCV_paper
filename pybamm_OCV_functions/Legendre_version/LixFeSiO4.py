import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -335008.968750 # G0 is the pure substance gibbs free energy 
    Omega0 = -136956.093750 
    Omega1 = -121303.773438 
    Omega2 = -16551.806641 
    Omega3 = 53115.562500 
    Omega4 = 9794.468750 
    Omega5 = 43438.316406 
    Omegas =[Omega0, Omega1, Omega2, Omega3, Omega4, Omega5]
    # phase boundary 0
    x_alpha_0 = 0.0000001000000012
    x_beta_0 = 0.0887559577822685
    mu_coex_0 = -575059.6875000000000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boundary 1
    x_alpha_1 = 0.5302119255065918
    x_beta_1 = 0.9764057397842407
    mu_coex_1 = -270381.9375000000000000
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion
    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)
    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0  + (1-is_outside_miscibility_gap_1)*mu_coex_1 )
    return -mu_e/96485.0



def legendre_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials up to degree n 
    using the Bonnet's recursion formula (i+1)P_(i+1)(x) = (2i+1)xP_i(x) - iP_(i-1)(x)
    and return all n functions in a list
    """
    # P = [torch.ones_like(x), x]  # P_0(x) = 1, P_1(x) = x
    P = [1.0, x]  # P_0(x) = 1, P_1(x) = x
    for i in range(1, n):
        P_i_plus_one = ((2 * i + 1) * x * P[i] - i * P[i - 1]) / (i + 1)
        P.append(P_i_plus_one)
    return P

def legendre_derivative_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials derivatives up to degree n 
    using (x^2-1)/n P'n(x) = xP_n(x) - P_(n-1)(x),
    and return all n functions in a list
    """
    Pn_values = legendre_poly_recurrence(x,n)
    Pn_derivatives = [0.0]
    for i in range(1, n+1):
        Pn_derivative_next = (x*Pn_values[i] - Pn_values[i-1])/((x**2-1)/i)
        Pn_derivatives.append(Pn_derivative_next)
    return Pn_derivatives

