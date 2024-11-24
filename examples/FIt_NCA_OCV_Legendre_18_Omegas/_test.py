from diffthermo.utils import write_ocv_functions
import torch.nn as nn
import numpy as np
import torch


Omega0 = nn.Parameter( torch.from_numpy(np.array([-30272.1758],dtype="float32")) )
Omega1 = nn.Parameter( torch.from_numpy(np.array([31340.5469],dtype="float32")) )
Omega2 = nn.Parameter( torch.from_numpy(np.array([9555.4678],dtype="float32")) )
Omega3 = nn.Parameter( torch.from_numpy(np.array([8178.6528],dtype="float32")) )
Omega4 = nn.Parameter( torch.from_numpy(np.array([-18701.7637],dtype="float32")) )
Omega5 = nn.Parameter( torch.from_numpy(np.array([785.7636],dtype="float32")) )
Omega6 = nn.Parameter( torch.from_numpy(np.array([-5834.5361],dtype="float32")) )
Omega7 = nn.Parameter( torch.from_numpy(np.array([11435.9912],dtype="float32")) )
Omega8 = nn.Parameter( torch.from_numpy(np.array([-1814.8590],dtype="float32")) )
Omega9 = nn.Parameter( torch.from_numpy(np.array([2022.1323 ],dtype="float32")) )
Omega10 = nn.Parameter( torch.from_numpy(np.array([-9657.7012],dtype="float32")) )
Omega11 = nn.Parameter( torch.from_numpy(np.array([-3034.7329 ],dtype="float32")) )
Omega12 = nn.Parameter( torch.from_numpy(np.array([-2728.6621 ],dtype="float32")) )
Omega13 = nn.Parameter( torch.from_numpy(np.array([5357.6255 ],dtype="float32")) )
Omega14 = nn.Parameter( torch.from_numpy(np.array([5190.5181 ],dtype="float32")) )
Omega15 = nn.Parameter( torch.from_numpy(np.array([6293.6187 ],dtype="float32")) )
Omega16 = nn.Parameter( torch.from_numpy(np.array([2882.0156 ],dtype="float32")) )
Omega17 = nn.Parameter( torch.from_numpy(np.array([1547.2567],dtype="float32")) )
G0 = nn.Parameter( torch.from_numpy(np.array([-380060.5000 ],dtype="float32")) )     

params_list = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8, Omega9, Omega10, Omega11, Omega12, Omega13, Omega14, Omega15, Omega16, Omega17, G0]


write_ocv_functions(params_list,polynomial_style = 'Legendre')



