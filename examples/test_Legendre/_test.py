from diffthermo.utils import write_ocv_functions
import torch.nn as nn
import numpy as np
import torch

Omega0 = nn.Parameter( torch.from_numpy(np.array([-7714.1719],dtype="float32")) )
Omega1 = nn.Parameter( torch.from_numpy(np.array([-8985.8242 ],dtype="float32")) )
Omega2 = nn.Parameter( torch.from_numpy(np.array([-12829.3418 ],dtype="float32")) )
Omega3 = nn.Parameter( torch.from_numpy(np.array([-4622.6899 ],dtype="float32")) )
Omega4 = nn.Parameter( torch.from_numpy(np.array([-8323.2715 ],dtype="float32")) )
Omega5 = nn.Parameter( torch.from_numpy(np.array([-2240.3699 ],dtype="float32")) )
Omega6 = nn.Parameter( torch.from_numpy(np.array([-3581.7393 ],dtype="float32")) )
Omega7 = nn.Parameter( torch.from_numpy(np.array([-1009.4918 ],dtype="float32")) )
Omega8 = nn.Parameter( torch.from_numpy(np.array([-1783.6764 ],dtype="float32")) )
G0 = nn.Parameter( torch.from_numpy(np.array([-11604.7490  ],dtype="float32")) )     

params_list = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, G0]


write_ocv_functions(params_list,polynomial_style = 'Legendre')



