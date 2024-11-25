import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from .energy import sampling, convex_hull, CommonTangent
# from .solver import CommonTangent


global  _eps
_eps = 1e-7


# loss function 
def collocation_loss_all_pts(mu, x, phase_boundarys_fixed_point, GibbsFunction, params_list, alpha_miscibility, T=300):
    """
    Calculate the collocation points loss for all datapoints (that way we don't need hessian loss and common tangent loss, everything is converted into collocation loss)
    mu is the measured OCV data times Farady constant
    x is the measured SOC data
    phase_boundarys_fixed_point is the list of starting and end point of miscibility gap(s)
    GibbsFunction is the Gibbs free energy landscape
    params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
    alpha_miscibility: weight of miscibility loss
    T: temperature
    """
    # see if x is in any gaps
    def _is_inside_gaps(_x, _gaps_list):
        _is_inside = False
        _index = -99999
        if len(_gaps_list) == 0:
            return False, -99999
        for i in range(0, len(_gaps_list)):
            if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                _is_inside = True
                _index = i
                break
        return _is_inside, _index
    # calculate loss
    loss_ = 0.0
    n_count = 0
    for i in range(0, len(x)):
        x_now = x[i]
        mu_now = mu[i]
        is_inside, index = _is_inside_gaps(x_now, phase_boundarys_fixed_point)
        if is_inside == False:
            # outside miscibility gap 
            x_now = x_now.requires_grad_()
            g_now = GibbsFunction(x_now, params_list, T)
            mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
            loss_ = loss_ + ((mu_pred_now-mu_now)/(8.314*T))**2 
            # print(x_now, mu_now, mu_pred_now)
            n_count = n_count + 1
        else: 
            # inside miscibility gap
            x_alpha = phase_boundarys_fixed_point[index][0]
            x_beta = phase_boundarys_fixed_point[index][1]
            ct_pred = (GibbsFunction(x_alpha, params_list, T) - GibbsFunction(x_beta, params_list, T))/(x_alpha - x_beta) 
            if torch.isnan(ct_pred):
                print("Common tangent is NaN")
                x_alpha = 99999.9
                x_beta = -99999.9
            if x_alpha > x_beta:
                print("Error in phase equilibrium boundary, x_left %.4f larger than x_right %.4f. If Hessian loss is not 0, it's fine. Otherwise check code carefully!" %(x_alpha, x_beta))
                x_alpha = 99999.9
                x_beta = -99999.9
            if torch.isnan(ct_pred):
                print("Warning: skipped for loss calculation at a filling fraction x")
            else:
                loss_ = loss_ + alpha_miscibility*((ct_pred - mu_now)/(8.314*T))**2
                # print(x_now, mu_now, ct_pred)
                n_count = n_count + 1
    return loss_/n_count





# train function
def train(datafile_name='graphite.csv', 
          number_of_Omegas=6, 
          polynomial_style = "Legendre",
          learning_rate = 1000.0, 
          total_training_epochs = 8000,
          loss_threshold = 0.01,
          G0_rand_range=[-10*5000,-5*5000], 
          Omegas_rand_range=[-10*100,10*100],
          records_y_lims = [0.0,0.6]):
    """
    Fit the diffthermo OCV function

    Inputs:
    datafile_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be OCV. Must not be header
    number_of_Omegas: number of R-K parameters. Note that the order of R-K expansion = number_of_Omegas - 1
    polynomial_style: style of polynomials to expand excess thermo, can be "Legendre", "R-K", "Chebyshev".
    learning_rate: learning rate for updating parameters
    total_training_epochs: total epochs for fitting
    loss_threshold: threshold of loss, below which training stops automatically
    G0_rand_range: the range for randomly initialize G0
    Omegas_rand_range: the range for randomly initialize R-K parameters
    records_y_lims: the range for records y axis lims

    Outputs: 
    params_list: contains fitted G0 and Omegas, which can be put into write_ocv_functions function to get your PyBaMM OCV function
    """
    
    if polynomial_style == "Legendre":
        from .energy import GibbsFE_Legendre as GibbsFE
    elif polynomial_style == "R-K":
        from .energy import GibbsFE_RK as GibbsFE
    elif polynomial_style == "Chebyshev":
        from .energy import GibbsFE_Chebyshev as GibbsFE
    
    working_dir = os.getcwd()
    os.chdir(working_dir)
    try:
        os.mkdir("records")
    except:
        pass
    with open("log",'w') as fin:
        fin.write("")

    # read data
    df = pd.read_csv(datafile_name,header=None)
    data = df.to_numpy()
    x = data[:,0]
    mu = -data[:,1]*96485 # because -mu_e- = OCV*F, -OCV*F = mu
    # convert to torch.tensor
    x = x.astype("float32")
    x = torch.from_numpy(x)
    mu = mu.astype("float32")
    mu = torch.from_numpy(mu)

    # declare all params
    params_list = []
    if number_of_Omegas <=10:
        for _ in range(0, number_of_Omegas):
            Omegai_start = np.random.randint(Omegas_rand_range[0], Omegas_rand_range[1])
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
    else:
        for _ in range(0, 10):
            Omegai_start = np.random.randint(Omegas_rand_range[0], Omegas_rand_range[1])
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
        for _ in range(10, number_of_Omegas):
            Omegai_start = 0.0
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
    G0_start = np.random.randint(G0_rand_range[0], G0_rand_range[1]) # G0 is the pure substance gibbs free energy 
    G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
    params_list.append(G0)
    params_list_for_optimizer = params_list

    # init optimizer
    optimizer = optim.Adam(params_list_for_optimizer, lr=learning_rate)

    # init loss weight
    alpha_collocation = 1.0 # deprecated, must be 1.0
    alpha_miscibility = 1.0 # deprecated, must be 1.0

    # train
    params_record = []
    for i in range(0, len(params_list)):
        params_record.append([])
    epoch_record = []
    loss = 9999.9 # init total loss
    epoch = -1
    while loss > loss_threshold and epoch < total_training_epochs:
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
            plt.ylim(records_y_lims)
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
    return params_list



def write_ocv_functions(params_list, polynomial_style = "R-K"):
    if polynomial_style == "Legendre":
        from .energy import GibbsFE_Legendre as GibbsFE
    elif polynomial_style == "R-K":
        from .energy import GibbsFE_RK as GibbsFE
    elif polynomial_style == "Chebyshev":
        from .energy import GibbsFE_Chebyshev as GibbsFE

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

    # print detected phase boundary
    cts = []
    if len(phase_boundary_fixed_point) > 0:
        print("Found %d phase coexistence region(s):" %(len(phase_boundary_fixed_point)))
        for i in range(0, len(phase_boundary_fixed_point)):
            x_alpha = phase_boundary_fixed_point[i][0]
            x_beta = phase_boundary_fixed_point[i][1]
            ct_now = (GibbsFE(x_alpha, params_list, T=300) - GibbsFE(x_beta, params_list, T=300))/(x_alpha - x_beta) 
            cts.append(ct_now)
            print("From x=%.16f to x=%.16f, mu_coex=%.16f" %(phase_boundary_fixed_point[i][0], phase_boundary_fixed_point[i][1], ct_now))
    else:
        print("No phase separation region detected.")

    # print output function in python
    with open("fitted_ocv_functions.py", "w") as fout:
        fout.write("import numpy as np\nimport pybamm\nfrom pybamm import exp, log, tanh, constants, Parameter, ParameterValues\n\n")
        fout.write("def fitted_OCP(sto):\n")
        fout.write("    _eps = 1e-7\n")
        fout.write("    # rk params\n")
        fout.write("    G0 = %.6f # G0 is the pure substance gibbs free energy \n" %(params_list[-1].item()))
        for i in range(0, len(params_list)-1):
            fout.write("    Omega%d = %.6f \n" %(i, params_list[i].item()))
        text = "    Omegas =["
        for i in range(0, len(params_list)-1):
            text=text+"Omega"+str(i)
            if i!= len(params_list)-2:
                text=text+", "
            else:
                text=text+"]\n"
        fout.write(text)
        # write phase boundaries
        if len(phase_boundary_fixed_point)>0:
            for i in range(0, len(phase_boundary_fixed_point)):
                fout.write("    # phase boundary %d\n" %(i))
                fout.write("    x_alpha_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][0]))
                fout.write("    x_beta_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][1]))
                fout.write("    mu_coex_%d = %.16f\n" %(i, cts[i]))
                fout.write("    is_outside_miscibility_gap_%d = (sto<x_alpha_%d) + (sto>x_beta_%d)\n" %(i,i,i))
            fout.write("    # whether is outside all gap\n")
            text = "    is_outside_miscibility_gaps = "
            for i in range(0, len(phase_boundary_fixed_point)):
                text = text + "is_outside_miscibility_gap_%d " %(i)
                if i!=len(phase_boundary_fixed_point)-1:
                    text = text + "* "
            fout.write(text)
            fout.write("    \n")
            fout.write("    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))\n")
            
            if polynomial_style == "R-K":
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))\n")
            elif polynomial_style == "Legendre":              
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")
            elif polynomial_style == "Chebyshev":
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
                fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Tn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Tn_values[i])\n")
            else:
                print("polynomial_style not recognized in write_ocv")
                exit()

            text0 = "    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   "
            text1 = ""
            for i in range(0, len(cts)):
                text1 = text1 + "(1-is_outside_miscibility_gap_%d)*mu_coex_%d " %(i, i)
                if i != len(cts)-1:
                    text1 = text1 + " + "
            text = text0 + "(" + text1 + ")\n"
            fout.write(text)
            fout.write("    return -mu_e/96485.0\n\n\n\n")
        else:
            # no phase boundaries required, just mu and return -mu/F
            fout.write("    mu = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))\n")
            if polynomial_style == "R-K":
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu = mu + Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))\n")
            elif polynomial_style == "Legendre":              
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu = mu -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")
            elif polynomial_style == "Chebyshev":
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
                fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(Omegas)):\n")
                fout.write("        mu = mu -2*sto*(1-sto)*(Omegas[i]*Tn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Tn_values[i])\n")
            fout.write("    return -mu/96485.0\n\n\n\n")
            
            
    if polynomial_style == "Legendre":
        abs_path = os.path.abspath(__file__)[:-8]+"__legendre_derivatives.py"
        with open(abs_path,'r') as fin:
            lines = fin.readlines()
        with open("fitted_ocv_functions.py", "a") as fout:
            for line in lines:
                fout.write(line)
    elif polynomial_style == "Chebyshev":
        abs_path = os.path.abspath(__file__)[:-8]+"__chebyshev_derivatives.py"
        with open(abs_path,'r') as fin:
            lines = fin.readlines()
        with open("fitted_ocv_functions.py", "a") as fout:
            for line in lines:
                fout.write(line)
            
    # print output function in matlab
    if polynomial_style == "R-K":
        with open("fitted_ocv_functions.m", "w") as fout:
            fout.write("function result = ocv(sto):\n")
            fout.write("    eps = 1e-7;\n")
            fout.write("    %% rk params\n")
            fout.write("    G0 = %.6f; %%G0 is the pure substance gibbs free energy \n" %(params_list[-1].item()))
            for i in range(0, len(params_list)-1):
                fout.write("    Omega%d = %.6f; \n" %(i, params_list[i].item()))
            text = "    Omegas =["
            for i in range(0, len(params_list)-1):
                text=text+"Omega"+str(i)
                if i!= len(params_list)-2:
                    text=text+", "
                else:
                    text=text+"];\n"
            fout.write(text)
            # write phase boundaries
            if len(phase_boundary_fixed_point)>0:
                for i in range(0, len(phase_boundary_fixed_point)):
                    fout.write("    %% phase boundary %d\n" %(i))
                    fout.write("    x_alpha_%d = %.16f ; \n" %(i, phase_boundary_fixed_point[i][0]))
                    fout.write("    x_beta_%d = %.16f ; \n" %(i, phase_boundary_fixed_point[i][1]))
                    fout.write("    mu_coex_%d = %.16f ; \n" %(i, cts[i]))
                    fout.write("    is_outside_miscibility_gap_%d = (sto<x_alpha_%d) + (sto>x_beta_%d) ; \n" %(i,i,i))
                fout.write("    %% whether is outside all gap\n")
                text = "    is_outside_miscibility_gaps = "
                for i in range(0, len(phase_boundary_fixed_point)):
                    text = text + "is_outside_miscibility_gap_%d " %(i)
                    if i!=len(phase_boundary_fixed_point)-1:
                        text = text + "* "
                fout.write(text)
                fout.write(";    \n")
                fout.write("    mu_outside = G0 + 8.314*300.0*log((sto+eps)/(1-sto+eps)) ; \n") 
                
                fout.write("    for i=0:length(Omegas)-1\n")
                fout.write("        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i+1]*((1-2*sto)^(i+1) - 2*i*sto*(1-sto)*(1-2*sto)^(i-1));\n")
                fout.write("end\n")
                
                text0 = "    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   "
                text1 = ""
                for i in range(0, len(cts)):
                    text1 = text1 + "(1-is_outside_miscibility_gap_%d)*mu_coex_%d " %(i, i)
                    if i != len(cts)-1:
                        text1 = text1 + " + "
                text = text0 + "(" + text1 + ") ;\n"
                fout.write(text)
                fout.write("    result = -mu_e/96485.0 ;")
                fout.write("    return;")
                fout.write("end  \n\n\n")
    else:
        print("Writing matlab ocv function only support R-K")
    # write complete
    print("\n\n\n\n\n Fitting Complete.\n")
    print("Fitted OCV function written in PyBaMM function (copy and paste readay!):\n")
    print("###################################\n")
    with open("fitted_ocv_functions.py", "r") as fin:
        lines = fin.readlines()
    for line in lines:
        print(line, end='')
    print("\n\n###################################\n")
    print("Or check fitted_ocv_functions.py and fitted_ocv_functions.m (if polynomial style = R-K) for fitted thermodynamically consistent OCV model in PyBaMM & Matlab formats. ")
