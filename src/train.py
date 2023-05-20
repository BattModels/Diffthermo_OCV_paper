import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt

from .diffthermo import GibbsFE, newton_raphson, sampling, convex_hull, FixedPointOperation, FixedPointOperationForwardPass, CommonTangent, collocation_loss_all_pts

# for debug only AMYAO DEBUG
global is_print , _eps
is_print = False
_eps = 1e-7

working_dir = os.getcwd()
try:
    os.mkdir("records")
except:
    pass

# read OCV data
df = pd.read_csv("YOUR_OCV_FILE.csv",header=None)  """ Put your OCV file name here """
data = df.to_numpy() """ first row shall be filling fraction of Li, second row shall be OCV"""
x = data[:,0]  """ Filling fraction of Li. Make sure the value of x is between 0 and 1 """
mu = -data[:,1]*96485 """ OCV times -96485 to convert OCV into chemical potential """# because -mu_e- = OCV*F, -OCV*F = mu 

# convert to torch.tensor
x = x.astype("float32")
x = torch.from_numpy(x)
mu = mu.astype("float32")
mu = torch.from_numpy(mu)
os.chdir(working_dir)
# init log
with open("log",'w') as fin:
    fin.write("")

# init params that wait for training 
""" Add as much Omegas as you like below """
G0_start = np.random.randint(-10,-5)*5000 # G0 is the pure substance gibbs free energy 
Omega0_start = np.random.randint(-10,10)*100
Omega1_start = np.random.randint(-10,10)*100
Omega2_start = np.random.randint(-10,10)*100
Omega3_start = np.random.randint(-10,10)*100
Omega4_start = np.random.randint(-10,10)*100
Omega5_start = np.random.randint(-10,10)*100
Omega6_start = np.random.randint(-10,10)*100
Omega7_start = np.random.randint(-10,10)*100
# convert them into torch variables
G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
Omega0 = nn.Parameter( torch.from_numpy(np.array([Omega0_start],dtype="float32")) ) 
Omega1 = nn.Parameter( torch.from_numpy(np.array([Omega1_start],dtype="float32")) ) 
Omega2 = nn.Parameter( torch.from_numpy(np.array([Omega2_start],dtype="float32")) ) 
Omega3 = nn.Parameter( torch.from_numpy(np.array([Omega3_start],dtype="float32")) ) 
Omega4 = nn.Parameter( torch.from_numpy(np.array([Omega4_start],dtype="float32")) ) 
Omega5 = nn.Parameter( torch.from_numpy(np.array([Omega5_start],dtype="float32")) ) 
Omega6 = nn.Parameter( torch.from_numpy(np.array([Omega6_start],dtype="float32")) ) 
Omega7 = nn.Parameter( torch.from_numpy(np.array([Omega7_start],dtype="float32")) ) 
""" Make sure you convert all Omegas and G0 into nn.Parameter """


# declare all params
""" Declare your Omegas and G0 here """
params_list = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, G0] 

# init optimizer
learning_rate = 1000.0 """ Change the learning rate as you like, 1000.0 is a recommended value but you can choose your own """


""" Declare your Omegas and G0 here for the optimizer, make sure it is the SAME as params_list """
params_list_for_optimizer = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, G0] 


optimizer = optim.Adam(params_list_for_optimizer, lr=learning_rate) 
# init loss weight, kept them as 1.0 (deprecated already)
alpha_collocation = 1.0
alpha_miscibility = 1.0 # miscibility region is less valued

# train
params_record = []
for i in range(0, len(params_list)):
    params_record.append([])
epoch_record = []
# total_epochs = 1000
# for epoch in range(0, total_epochs):
loss = 9999.9 # init total loss
total_epoch_num = 8000 """ Change total number of epoch here """
epoch = -1
while loss > 0.0001 and epoch < total_epoch_num:
    # clean grad info
    optimizer.zero_grad()
    # use current params to calculate predicted phase boundary
    epoch = epoch + 1
    # init loss components
    loss = 0.0 # init total loss
    loss_hessian = 0.0 # hessian loss in case no phase boundary is found
    loss_collocation = 0.0 # collocation points loss
    loss_fake_gap = 0.0 # penalizing the redundant miscibility gap(s)
    # sample the Gibbs free energy landscape
    sample = sampling(GibbsFE, params_list, T=300, sampling_id=1)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is at least one phase boundary predicted 
        phase_boundary_fixed_point = []
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, params_list, T = 300) # init common tangent model
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
            phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
    else:
        # No boundary find.
        phase_boundary_fixed_point = []
    loss_collocation = alpha_collocation * collocation_loss_all_pts(mu, x, phase_boundary_fixed_point, GibbsFE, params_list, alpha_miscibility, T=300)
    # backprop
    loss = loss_collocation*1.0
    loss.backward()
    optimizer.step()
    # record
    for i in range(0, len(params_list)):
        params_record[i].append(params_list[i].item()/1000.0)
    epoch_record.append(epoch)
    # print output
    output_txt = "Epoch %3d  Loss %.4f     " %(epoch, loss)
    for i in range(0, len(params_list)-1):
        output_txt = output_txt + "Omega%d %.4f "%(i, params_list[i].item())
    output_txt  = output_txt + "G0 %.4f "%(params_list[-1].item())
    output_txt = output_txt + "      "
    # output_txt = output_txt + "x_left %.4f "%(phase_boundary_fixed_point[0].item())
    # output_txt = output_txt + "x_right %.4f "%(phase_boundary_fixed_point[1].item())
    print(output_txt)
    with open("log",'a') as fin:
        fin.write(output_txt)
        fin.write("\n")
    # check training for every 100 epochs
    if epoch % 100 == 0:
        # # draw the fitted results
        mu_pred = []
        for i in range(0, len(x)):
            x_now = x[i]
            mu_now = mu[i]
            x_now = x_now.requires_grad_()
            g_now = GibbsFE(x_now, params_list, T=300)
            mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
            mu_pred.append(mu_pred_now.detach().numpy())
        mu_pred = np.array(mu_pred)
        SOC = x.clone().numpy()
        # plot figure
        plt.figure(figsize=(5,4))
        # plot the one before common tangent construction
        U_pred_before_ct = mu_pred/(-96485)
        plt.plot(SOC, U_pred_before_ct, 'k--', label="Prediction Before CT Construction")
        # plot the one after common tangent construction
        mu_pred_after_ct = []
        # see if x is inside any gaps
        def _is_inside_gaps(_x, _gaps_list):
            _is_inside = False
            _index = -99999
            for i in range(0, len(_gaps_list)):
                if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                    _is_inside = True
                    _index = i
                    break
            return _is_inside, _index
        # pred
        for i in range(0, len(x)):
            x_now = x[i]
            mu_now = mu[i]
            is_inside, index = _is_inside_gaps(x_now, phase_boundary_fixed_point)
            if is_inside == False:
                # outside miscibility gap 
                mu_pred_after_ct.append(mu_pred[i])
            else: 
                # inside miscibility gap
                x_alpha = phase_boundary_fixed_point[index][0]
                x_beta = phase_boundary_fixed_point[index][1]
                ct_pred = (GibbsFE(x_alpha, params_list, T=300) - GibbsFE(x_beta, params_list, T=300))/(x_alpha - x_beta) 
                if torch.isnan(ct_pred) == False:
                    mu_pred_after_ct.append(ct_pred.clone().detach().numpy()[0]) 
                else:
                    mu_pred_after_ct.append(mu_pred[i])
        mu_pred_after_ct = np.array(mu_pred_after_ct)
        U_pred_after_ct = mu_pred_after_ct/(-96485)
        plt.plot(SOC, U_pred_after_ct, 'r-', label="Prediction After CT Construction")
        U_true_value = mu.numpy()/(-96485) # plot the true value
        plt.plot(SOC, U_true_value, 'b-', label="True OCV")
        plt.xlim([0,1])
        plt.ylim([0.0, 0.6])
        plt.legend()
        plt.xlabel("SOC")
        plt.ylabel("OCV")
        fig_name = "At_Epoch_%d.png" %(epoch)
        os.chdir("records")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()
        os.chdir("../")
        # # draw the RK params VS epochs
        total_epochs = len(epoch_record)
        for i in range(0, len(params_list)-1):
            plt.figure(figsize=(5,4))
            param_name = "Omega%d" %(i)
            plt.plot(epoch_record, params_record[i], 'r-', label=param_name)
            plt.xlim([0,total_epochs])
            plt.xlabel("Epoch")
            plt.ylabel("Param")
            plt.legend()
            fig_name = param_name+".png"
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close()
        # # draw G0 VS epochs
        plt.figure(figsize=(5,4))
        plt.plot(epoch_record, params_record[-1], 'r-', label="G0")
        plt.xlim([0,total_epochs])
        plt.xlabel("Epoch")
        plt.ylabel("Param")
        plt.legend()
        fig_name = "G0.png" 
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()


print("Training Complete.\n")
exit()