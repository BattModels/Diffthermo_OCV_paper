import numpy as np
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

def fitted_OCP(sto):
    _eps = 1e-7
    # rk params
    G0 = -155433.015625 # G0 is the pure substance gibbs free energy 
    Omega0 = -36209.914062 
    Omega1 = -4201.322266 
    Omega2 = -50238.488281 
    Omega3 = -6594.017090 
    Omegas =[Omega0, Omega1, Omega2, Omega3]
    # phase boundary 0
    x_alpha_0 = 0.1186181306838989
    x_beta_0 = 0.8482595682144165
    mu_coex_0 = -155769.4218750000000000
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0     
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion
    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)
    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Tn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Tn_values[i])
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   ((1-is_outside_miscibility_gap_0)*mu_coex_0 )
    return -mu_e/96485.0



def chebyshev_poly_recurrence(x, n):
    """
    Compute the Chebyshev polynomials (first kind) up to degree n 
    using the recursion formula T_(n+1)(x) = 2xT_n(x) - T_(n-1)(x),
    and return all n functions in a list
    """
    # T = [torch.ones_like(x), x]  # T_0(x) = 1, T_1(x) = x
    T = [1.0, x]
    for i in range(1, n):
        T_i_plus_1 = 2*x*T[i] - T[i-1]
        T.append(T_i_plus_1)
    return T


def chebyshev_2nd_kindpoly_recurrence(x, n):
    """
    Compute the Chebyshev polynomials (second kind) up to degree n 
    using the recursion formula U_(n+1)(x) = 2xU_n(x) - U_(n-1)(x),
    and return all n functions in a list
    """
    # U = [torch.ones_like(x), 2*x]  # U_0(x) = 1, U_1(x) = 2x
    U = [1.0, 2*x]
    for i in range(1, n):
        U_i_plus_1 = 2*x*U[i] - U[i-1]
        U.append(U_i_plus_1)
    return U

def chebyshev_derivative_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials derivatives up to degree n 
    using T'n(x) = n*U_(n-1)(x),
    and return all n functions in a list
    """
    Un_values = chebyshev_2nd_kindpoly_recurrence(x,n)
    Tn_derivatives = [0.0]
    for i in range(1, n+1):
        Tn_derivatives_next = i*Un_values[i-1]
        Tn_derivatives.append(Tn_derivatives_next)
    return Tn_derivatives

