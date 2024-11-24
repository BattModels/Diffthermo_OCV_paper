import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(-1+0.01,1-0.01,201)

# https://en.wikipedia.org/wiki/Chebyshev_polynomials#Examples
T0 = 1 * (1-x**2)**(-0.25) # *(-x+1)*(x+1)
T1 = x * (1-x**2)**(-0.25) # *(-x+1)*(x+1)
T2 = (2*x**2 - 1) * (1-x**2)**(-0.25) # *(-x+1)*(x+1)
T3 = (4*x**3 - 3*x) * (1-x**2)**(-0.25) # *(-x+1)*(x+1)
T4 = (8*x**4 - 8*x**2 +1) * (1-x**2)**(-0.25) # *(-x+1)*(x+1)
T5 = (16*x**5 - 20*x**3 +5*x) * (1-x**2)**(-0.25) # *(-x+1)*(x+1)

plt.plot(x, T0, label="0")
plt.plot(x, T1, label="1")
plt.plot(x, T2, label="2")
plt.plot(x, T3, label="3")
plt.plot(x, T4, label="4")
plt.plot(x, T5, label="5")
plt.legend()
# plt.show()
plt.ylim([-2.5,2.5])
plt.savefig("normalized_Chebyshev.png")

