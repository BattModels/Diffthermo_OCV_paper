import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # Legendre rk params
    G0 = -380060.500000 # G0 is the pure substance gibbs free energy 
    Omega0 = -30272.175781 
    Omega1 = 31340.546875 
    Omega2 = 9555.467773 
    Omega3 = 8178.652832 
    Omega4 = -18701.763672 
    Omega5 = 785.763611 
    Omega6 = -5834.536133 
    Omega7 = 11435.991211 
    Omega8 = -1814.859009 
    Omega9 = 2022.132324 
    Omega10 = -9657.701172 
    Omega11 = -3034.732910 
    Omega12 = -2728.662109 
    Omega13 = 5357.625488 
    Omega14 = 5190.518066 
    Omega15 = 6293.618652 
    Omega16 = 2882.015625 
    Omega17 = 1547.256714 
    Omegas =[Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, Omega9, Omega10, Omega11, Omega12, Omega13, Omega14, Omega15, Omega16, Omega17]
    # phase boundary 0
    x_alpha_0 = 0.0002851486206055
    x_beta_0 = 0.3705533742904663
    mu_coex_0 = -406251.2812500000000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion
    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)
    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0 )
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
