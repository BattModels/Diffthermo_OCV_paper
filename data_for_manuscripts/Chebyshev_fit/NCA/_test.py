from diffthermo.utils import write_ocv_functions
import torch.nn as nn
import numpy as np
import torch

Omega0 = nn.Parameter( torch.from_numpy(np.array([-35820.7227 ],dtype="float32")) )
Omega1 = nn.Parameter( torch.from_numpy(np.array([29697.7344],dtype="float32")) )
Omega2 = nn.Parameter( torch.from_numpy(np.array([-1744.3585 ],dtype="float32")) )
Omega3 = nn.Parameter( torch.from_numpy(np.array([8340.0947 ],dtype="float32")) )
Omega4 = nn.Parameter( torch.from_numpy(np.array([-11276.0078 ],dtype="float32")) )
Omega5 = nn.Parameter( torch.from_numpy(np.array([1330.2571 ],dtype="float32")) )
Omega6 = nn.Parameter( torch.from_numpy(np.array([-7174.8413 ],dtype="float32")) )
Omega7 = nn.Parameter( torch.from_numpy(np.array([2218.0413 ],dtype="float32")) )
Omega8 = nn.Parameter( torch.from_numpy(np.array([-1576.3016],dtype="float32")) )
Omega9 = nn.Parameter( torch.from_numpy(np.array([4413.5649 ],dtype="float32")) )
Omega10 = nn.Parameter( torch.from_numpy(np.array([2054.7432 ],dtype="float32")) )
Omega11 = nn.Parameter( torch.from_numpy(np.array([3666.9746 ],dtype="float32")) )
Omega12 = nn.Parameter( torch.from_numpy(np.array([1336.4106 ],dtype="float32")) )
Omega13 = nn.Parameter( torch.from_numpy(np.array([1056.8680 ],dtype="float32")) )
G0 = nn.Parameter( torch.from_numpy(np.array([-380637.6875 ],dtype="float32")) )

params_list = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, Omega9, Omega10, Omega11, Omega12, Omega13, G0]


write_ocv_functions(params_list,polynomial_style = 'Chebyshev')



