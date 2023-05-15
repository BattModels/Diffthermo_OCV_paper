import os
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
   

# for debug only AMYAO DEBUG
global is_print , _eps
is_print = False
_eps = 1e-7


def GibbsExcess(x, params_list, order_list):
    G = 0.0
    for i in range(0, len(params_list)):
        G = G + x*(1-x)*(params_list[i]*(1-2*x)**order_list[i])
    return G


x = torch.from_numpy(np.linspace(0.01,0.99,99))

# init params that wait for training 
G0_start = -336668.3750 # G0 is the pure substance gibbs free energy 
Omega0_start = 128205.6641
Omega1_start = 61896.4375
Omega2_start = -290954.8125
Omega3_start = -164259.3750
G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
Omega0 = nn.Parameter( torch.from_numpy(np.array([Omega0_start],dtype="float32")) ) 
Omega1 = nn.Parameter( torch.from_numpy(np.array([Omega1_start],dtype="float32")) ) 
Omega2 = nn.Parameter( torch.from_numpy(np.array([Omega2_start],dtype="float32")) ) 
Omega3 = nn.Parameter( torch.from_numpy(np.array([Omega3_start],dtype="float32")) ) 

# draw the \Omega * x * (1-x) term in Gibbs free energy with all terms
G_excess = GibbsExcess(x, [Omega0, Omega1, Omega2, Omega3], [0,1,2,3])
G_excess = G_excess.detach().numpy()
x_plot = x.numpy()
plt.plot(x_plot, G_excess, label = "Total Excess")
# # draw the \Omega * x * (1-x) term in Gibbs free energy with even terms
# G_excess_even = GibbsExcess(x, [Omega0, Omega2], [0,2])
# G_excess_even = G_excess_even.detach().numpy()
# x_plot = x.numpy()
# plt.plot(x_plot, G_excess_even, label = "Even Terms in Excess")
# # draw the \Omega * x * (1-x) term in Gibbs free energy with odd terms
# G_excess_odd = GibbsExcess(x, [Omega1, Omega3], [1,3])
# G_excess_odd = G_excess_odd.detach().numpy()
# x_plot = x.numpy()
# plt.plot(x_plot, G_excess_odd, label = "Odd Terms in Excess")
# draw the \Omega0 * x * (1-x) term in Gibbs free energy 
G_excess_0 = GibbsExcess(x, [Omega0], [0])
G_excess_0 = G_excess_0.detach().numpy()
x_plot = x.numpy()
plt.plot(x_plot, G_excess_0, label = "Zeroth term in Excess")
plt.xlabel("Mole Fraction of Li")
plt.ylabel("Gibbs Free Energy")
plt.xlim([0,1])
plt.ylim([-10000, 50000])
plt.legend()
plt.savefig("Excess_discharge.jpg")


