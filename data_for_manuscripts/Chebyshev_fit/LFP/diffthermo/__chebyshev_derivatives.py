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

