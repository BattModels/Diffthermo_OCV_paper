import numpy as np
# import pybamm
# from pybamm import exp, log, tanh, constants, Parameter, ParameterValues
from numpy import log
import matplotlib.pyplot as plt

def graphite_OCP(sto):
    """
    diffthermo fit for graphite
    sto: stochiometry 
    """
    _eps = 1e-7
    # legendre rk params
    Omega0 = -7714.1719 
    Omega1 = -8985.8242 
    Omega2 = -12829.3418 
    Omega3 = -4622.6899 
    Omega4 = -8323.2715 
    Omega5 = -2240.3699 
    Omega6 = -3581.7393 
    Omega7 = -1009.4918 
    Omega8 = -1783.6764 
    G0 = -11604.7490       

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8]

    # phase boudary 0
    x_alpha_0 = 0.5630064010620117
    x_beta_0 = 0.8225407600402832
    mu_coex_0 = -7809.7387695312500000 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.2833885550498962
    x_beta_1 = 0.4296195507049561
    mu_coex_1 = -11121.4472656250000000 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*np.log((sto+_eps)/(1-sto+_eps))
    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion
    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)
    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1
    print(len(Pn_values), len(Pn_derivatives_values),len(Omegas))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
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

# Function to compute Legendre polynomials using recurrence relation
def legendre_derivative_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials up to degree n 
    using (x^2-1)/n P'n(x) = xP_n(x) - P_(n-1)(x),
    and return all n functions in a list
    """
    Pn_values = legendre_poly_recurrence(x,n)
    Pn_derivatives = [0.0]
    for i in range(1, n+1):
        Pn_derivative_next = (x*Pn_values[i] - Pn_values[i-1])/((x**2-1)/i)
        Pn_derivatives.append(Pn_derivative_next)
    return Pn_derivatives


x = np.linspace(0.01,0.99,999)
y = graphite_OCP(x)
plt.plot(x,y)
plt.show()


