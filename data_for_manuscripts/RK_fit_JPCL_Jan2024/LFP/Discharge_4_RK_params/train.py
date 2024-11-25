import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt

# for debug only AMYAO DEBUG
global is_print , _eps
is_print = False
_eps = 1e-7

try:
    os.mkdir("records")
except:
    pass


"""
The basic theory behind:
For an intercalation reaction Li+ + e- + HM = Li-HM (HM is host material), we have
mu_Li+ + mu_e- + mu_HM = mu_Li-HM
while  mu_e- = -N_A*e*OCV = -F*OCV (OCV is the open circut voltage), 
and mu_Li-HM can be expressed with the gibbs free energy for mixing (pure+ideal+excess), we have
-mu_e- = F*OCV = mu_Li+ + mu_HM - mu_Li-HM
i.e. 
mu_e- = -F*OCV = -mu_Li+ - mu_HM + mu_Li-HM 
We want to fit G_guess via diff thermo such that mu_guess = dG_guess/dx = mu_e- = -F*OCV
"""
"""
For OCV curves, since F*OCV = - mu_e-, the stability criteria according to 2nd law of thermo is dmu_e-/dx > 0, i.e.  d OCV / dx < 0, 
i.e. whenever d OCV /dx >0, this region is unstable.
"""

"""
We have G_Li+ + G_e- + G_HM = G_Li-HM
while G_Li-HM = x*G0 + (1-x)*G1 + R*T*(x*ln(x) + (1-x)*ln(1-x)) + L0*x*(1-x)
Therefore
G_e- = G_Li-HM - G_Li+ - G_HM = (redefine a ref state) x*G0 + (1-x)*0 + R*T*(x*ln(x) + (1-x)*ln(1-x)) + L0*x*(1-x)
"""


def GibbsFE(x, params_list, T = 300):
    """
    Expression for Gibbs Free Energy of charging / discharging process
    params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
    T is temperature, default value is 300K
    """
    G0 = params_list[-1]
    x = torch.clamp(x, min=_eps, max=1.0-_eps)
    G = x*G0 + (1-x)*0 + 8.314*T*(x*torch.log(x)+(1-x)*torch.log(1-x)) 
    for i in range(0, len(params_list)-1):
        G = G + x*(1-x)*(params_list[i]*(1-2*x)**i)
    return G


def newton_raphson(func, x0, threshold=1e-6, in_backward_hood = False):
    """
    x0: initial guess, with shape torch.Size([2])
    """
    error = 9999999.9
    x_now = x0.clone()
    # define g function for Newton-Raphson
    def g(x):
        return func(x) - x
    # iteration
    n_iter = -1
    while error > threshold and n_iter < 1000:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        f_now = g(x_now)
        J = autograd.functional.jacobian(g, x_now)
        f_now = torch.reshape(f_now, (2,1)) 
        x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (2,)) 
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp
        x_new[0] = torch.max(torch.tensor([0.0+_eps, x_new[0]]))
        x_new[1] = torch.min(torch.tensor([1.0-_eps, x_new[1]])) # +- 1e-6 is for the sake of torch.log. We don't want log function to blow up at x=0!
        x_now = x_now.clone().detach() # detach for memory saving
        # calculate error
        if torch.abs(x_new[0]-x_now[0]) < torch.abs(x_new[1]-x_now[1]):
            error = torch.abs(x_new[0]-x_now[0])
        else:
            error = torch.abs(x_new[1]-x_now[1])
        # step forward
        x_now = x_new.clone()
        n_iter = n_iter + 1
    if n_iter >= 999:
        print("Warning: Max iteration in Newton-Raphson solver reached.")
    return x_now


def sampling(GibbsFunction, params_list, T, sampling_id, ngrid=99, requires_grad = False):
    """
    Sampling a Gibbs free energy function (GibbsFunction)
    sampling_id is for recognition, must be a interger
    """
    x = np.concatenate((np.array([_eps]),np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),np.array([1.0-_eps]))) 
    x = torch.from_numpy(x.astype("float32"))
    x = x.requires_grad_()
    sample = torch.tensor([[x[i], GibbsFunction(x[i], params_list, T), sampling_id] for i in range(0, len(x))])
    return sample


def convex_hull(sample, ngrid=99, tolerance = _eps):
    """
    Convex Hull Algorithm that provides the initial guess for common tangent
    Need NOT to be differentiable
    returning the initial guess for common tangent & corresponding phase id
    Adapted from Pinwe's Jax-Thermo with some modifications
    Cite Pinwen's Jax-TherMo!
    """
    # convex hull, starting from the furtest points at x=0 and 1 and find all pieces
    base = [[sample[0,:], sample[-1,:]]]
    current_base_length = len(base) # currently len(base) = 1
    new_base_length = 9999999
    base_working = base.copy()
    n_iter = 0
    while new_base_length != current_base_length:
        n_iter = n_iter + 1
        # save historical length of base, for comparison at the end
        current_base_length = len(base)
        # continue the convex hull pieces construction until we find all pieces
        base_working_new=base_working.copy()
        for i in range(len(base_working)):   # len(base_working) = 1 at first, but after iterations on n, the length of this list will be longer
            # the distance of sampling points to the hyperplane formed by base vector
            # sample[:,column]-h[column] calculates the x and y distance for all sample points to the base point h
            # 0:2 deletes the sampling_id
            # t[column]-h[column] is the vector along the hyperplane (line in 2D case)
            # dot product of torch.tensor([[0.0,-1.0],[1.0,0.0]]) and t[column]-h[column] calculates the normal vector of the hyperplane defined by t[column]-h[column]
            h = base_working[i][0]; t = base_working[i][1] # h is the sample point at left side, t is the sample point at right side
            _n = torch.matmul(torch.from_numpy(np.array([[0.0,-1.0],[1.0,0.0]]).astype("float32")), torch.reshape((t[0:2]-h[0:2]), (2,1)))
            # limit to those having x value between the x value of h and t
            left_id = torch.argmin(torch.abs(sample[:,0]-h[0])) + 1 # limiting the searching range within h and t
            right_id = torch.argmin(torch.abs(sample[:,0]-t[0]))
            if left_id == right_id: # it means this piece of convex hull is the shortest piece possible
                base_working_new.remove(base_working[i])
            else:
                # it means it's still possible to make this piece of convex hull shorter
                sample_current = sample[left_id:right_id, :] 
                _t = sample_current[:,0:2]-h[0:2]
                dists = torch.matmul(_t, _n).squeeze()
                if dists.shape == torch.Size([]): # in case that there is only 1 item in dists, .squeeze wil squeeze ALL dimension and make dists a 0-dim tensor
                    dists = torch.tensor([dists])
                # select those underneath the hyperplane
                outer = []
                for _ in range(0, sample_current.shape[0]):
                    if dists[_] < -_eps: 
                        outer.append(sample_current[_,:]) 
                # if there are points underneath the hyperplane, select the farthest one. If no outer points, then this set of working base is dead
                if len(outer):
                    pivot = sample_current[torch.argmin(dists)] # the furthest node below the hyperplane defined hy t[column]-h[column]
                    # after find the furthest node, we remove the current hyperplane and rebuild two new hyperplane
                    z = 0
                    while (z<=len(base)-1):
                        # i.e. finding the plane corresponding to the current working plane
                        diff = torch.max(  torch.abs(torch.cat((base[z][0], base[z][1])) - torch.cat((base_working[i][0], base_working[i][1])))  )
                        if diff < tolerance:
                            # remove this plane
                            base.pop(z) # The pop() method removes the item at the given index from the list and returns the removed item.
                        else:
                            z=z+1
                    # the furthest node below the hyperplane is picked up to build two new facets with the two corners 
                    base.append([h, pivot])
                    base.append([pivot, t])
                    # update the new working base
                    base_working_new.remove(base_working[i])
                    base_working_new.append([h, pivot])
                    base_working_new.append([pivot, t])
                else:
                    base_working_new.remove(base_working[i])
        base_working=base_working_new
        # update length of base
        new_base_length = len(base)
    # find the pieces longer than usual. If for a piece of convex hull, the length of it is longer than delta_x
    delta_x = 1.0/(ngrid+1.0) + tolerance
    miscibility_gap_x_left_and_right = []
    miscibility_gap_phase_left_and_right = []
    for i in range(0, len(base)):
        convex_hull_piece_now = base[i]
        if convex_hull_piece_now[1][0]-convex_hull_piece_now[0][0] > delta_x:
            miscibility_gap_x_left_and_right.append(torch.tensor([convex_hull_piece_now[0][0], convex_hull_piece_now[1][0]]))
            miscibility_gap_phase_left_and_right.append(torch.tensor([convex_hull_piece_now[0][2], convex_hull_piece_now[1][2]]))
    return miscibility_gap_x_left_and_right, miscibility_gap_phase_left_and_right    


class FixedPointOperation(nn.Module):
    def __init__(self, G, params_list, T = 300):
        """
        The fixed point operation used in the backward pass of common tangent approach. 
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
        """
        super(FixedPointOperation, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        x_alpha = x[0]
        x_beta = x[1]
        g_right = self.G(x_beta, params_list, self.T) 
        g_left = self.G(x_alpha, params_list, self.T)
        mu_right = autograd.grad(outputs=g_right, inputs=x_beta, create_graph=True)[0]
        mu_left = autograd.grad(outputs=g_left, inputs=x_alpha, create_graph=True)[0]
        x_alpha_new = x_beta - (g_right - g_left)/(mu_left + _eps)
        x_alpha_new = torch.clamp(x_alpha_new , min=0.0+_eps, max=1.0-_eps) # clamp
        x_alpha_new = x_alpha_new.reshape(1)
        x_beta_new = x_alpha + (g_right - g_left)/(mu_right + _eps)
        x_beta_new = torch.clamp(x_beta_new , min=0.0+_eps, max=1.0-_eps) # clamp
        x_beta_new = x_beta_new.reshape(1)
        return torch.cat((x_alpha_new, x_beta_new))


# In case that the above implementation doesn't work
class FixedPointOperationForwardPass(nn.Module):
    def __init__(self, G, params_list, T = 300):
        """
        The fixed point operation used in the forward pass of common tangent approach
        Here we don't use the above implementation (instead we use Pinwen's implementation in Jax-TherMo) to guarantee that the solution converges to the correct places in forward pass
        G is the Gibbs free energy function 
        params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
        """
        super(FixedPointOperationForwardPass, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        # x_alpha_0 = (x[0]).reshape(1)
        # x_beta_0 = (x[1]).reshape(1)
        x_alpha_now = x[0]
        x_beta_now = x[1]
        x_alpha_now = x_alpha_now.reshape(1)
        x_beta_now = x_beta_now.reshape(1)
        g_left = self.G(x_alpha_now, params_list, self.T)
        g_right = self.G(x_beta_now, params_list, self.T)
        common_tangent = (g_left - g_right)/(x_alpha_now - x_beta_now)
        dcommon_tangent = 9999999.9
        n_iter_ct = 0
        while dcommon_tangent>1e-4 and n_iter_ct < 300:  
            """
            eq1 & eq2: we want dG/dx evaluated at x1 (and x2) to be the same as common_tangent, i.e. mu(x=x1 or x2) = common_tangent
            then applying Newton-Rapson iteration to solve f(x) = mu(x) - common_tangent, where mu(x) = dG/dx
            Newton-Rapson iteration: x1 = x0 - f(x0)/f'(x0)
            """  
            def eq(x):
                y = self.G(x, params_list, self.T) - common_tangent*x
                return y
            # update x_alpha
            dx = torch.tensor(999999.0)
            n_iter_dxalpha = 0.0
            while torch.abs(dx) > 1e-6 and n_iter_dxalpha < 300:
                x_alpha_now = x_alpha_now.requires_grad_()
                value_now = eq(x_alpha_now)
                f_now = autograd.grad(value_now, x_alpha_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_alpha_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_alpha_now = x_alpha_now + dx
                x_alpha_now = x_alpha_now.clone().detach()
                # clamp
                x_alpha_now = torch.max(torch.tensor([0.0+_eps, x_alpha_now]))
                x_alpha_now = torch.min(torch.tensor([1.0-_eps, x_alpha_now])) 
                x_alpha_now = x_alpha_now.reshape(1)
                n_iter_dxalpha = n_iter_dxalpha + 1
            # update x_beta
            dx = torch.tensor(999999.0)
            n_iter_dxbeta = 0.0
            while torch.abs(dx) > 1e-6 and n_iter_dxbeta < 300:
                x_beta_now = x_beta_now.requires_grad_()
                value_now = eq(x_beta_now)
                f_now = autograd.grad(value_now, x_beta_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_beta_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_beta_now = x_beta_now + dx
                x_beta_now = x_beta_now.clone().detach()
                # clamp
                x_beta_now = torch.max(torch.tensor([0.0+_eps, x_beta_now]))
                x_beta_now = torch.min(torch.tensor([1.0-_eps, x_beta_now])) 
                x_beta_now = x_beta_now.reshape(1)
                n_iter_dxbeta = n_iter_dxbeta + 1
            # after getting new x1 and x2, calculates the new common tangent, the same process goes on until the solution is self-consistent
            common_tangent_new = (self.G(x_alpha_now, params_list, self.T) - self.G(x_beta_now, params_list, self.T))/(x_alpha_now - x_beta_now)
            dcommon_tangent = torch.abs(common_tangent_new-common_tangent)
            common_tangent = common_tangent_new.clone().detach()
            n_iter_ct = n_iter_ct + 1
        return torch.cat((x_alpha_now, x_beta_now))


class CommonTangent(nn.Module):
    """
    Common Tangent Approach for phase equilibrium boundary calculation
    """
    def __init__(self, G, params_list, T = 300):
        super(CommonTangent, self).__init__()
        self.f_forward = FixedPointOperationForwardPass(G, params_list, T) # define forward operation here
        self.f = FixedPointOperation(G, params_list, T) # define backward operation here
        self.solver = newton_raphson
        self.f_thres = 1e-6
        self.T = T
    def forward(self, x, **kwargs):
        """
        x is the initial guess provided by convex hull
        """
        # Forward pass
        x_star = self.solver(self.f, x, threshold=self.f_thres) # use newton-raphson to get the fixed point
        if torch.any(torch.isnan(x_star)) == True: # in case that the previous one doesn't work
            print("Fixpoint solver failed at T = %d. Use traditional approach instead" %(self.T))
            x_star = self.f_forward(x)
        # (Prepare for) Backward pass
        new_x_star = self.f(x_star.requires_grad_()) # go through the process again to get derivative
        # register hook, can do anything with the grad that passed in
        def backward_hook(grad):
            # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
            if self.hook is not None:
                self.hook.remove()
                # torch.cuda.synchronize()   # To avoid infinite recursion
            """
            Compute the fixed point of y = yJ + grad, 
            where y is the new_grad, 
            J=J_f is the Jacobian of f at z_star, 
            grad is the input from the chain rule.
            From y = yJ + grad, we have (I-J)y = grad, so y = (I-J)^-1 grad
            """
            # # Original implementation by Shaojie Bai:
            # new_grad = self.solver(lambda y: autograd.grad(new_x_star, x_star, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), threshold=self.f_thres, in_backward_hood=True)
            # AM Yao: use inverse jacobian
            I_minus_J = torch.eye(2) - autograd.functional.jacobian(self.f, x_star)
            new_grad = torch.linalg.pinv(I_minus_J)@grad
            return new_grad
        # hook registration
        self.hook = new_x_star.register_hook(backward_hook)
        # all set! return 
        return new_x_star


def collocation_loss_all_pts(mu, x, phase_boundary_fixed_point, GibbsFunction, params_list, alpha_miscibility, T=300):
    """
    Calculate the collocation points loss for all datapoints (that way we don't need hessian loss and common tangent loss, everything is converted into collocation loss)
    mu is the OCV data
    x is the SOC data
    phase_boundary_fixed_point is the starting and end point of miscibility gap
    GibbsFunction is the Gibbs free energy landscape
    params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
    alpha_miscibility: weight of miscibility loss
    T: temperature
    """
    loss_ = 0.0
    x_alpha = phase_boundary_fixed_point[0]
    x_beta = phase_boundary_fixed_point[1]
    ct_pred = (GibbsFunction(x_alpha, params_list, T) - GibbsFunction(x_beta, params_list, T))/(x_alpha - x_beta) 
    if torch.isnan(ct_pred):
        print("Common tangent is NaN")
        x_alpha = 99999.9
        x_beta = -99999.9
    if x_alpha > x_beta:
        print("Error in phase equilibrium boundary, x_left %.4f larger than x_right %.4f. If Hessian loss is not 0, it's fine. Otherwise check code carefully!" %(x_alpha, x_beta))
        x_alpha = 99999.9
        x_beta = -99999.9
    # calculate loss
    n_count = 0
    for i in range(0, len(x)):
        x_now = x[i]
        mu_now = mu[i]
        if x_now < x_alpha or x_now > x_beta:
            # outside miscibility gap 
            x_now = x_now.requires_grad_()
            g_now = GibbsFunction(x_now, params_list, T)
            mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
            loss_ = loss_ + ((mu_pred_now-mu_now)/(8.314*T))**2 
            # print(x_now, mu_now, mu_pred_now)
            n_count = n_count + 1
        else: 
            # inside miscibility gap
            if torch.isnan(ct_pred):
                pass
            else:
                loss_ = loss_ + alpha_miscibility*((ct_pred - mu_now)/(8.314*T))**2
                # print(x_now, mu_now, ct_pred)
                n_count = n_count + 1
    return loss_/n_count





# read hysterisis data
working_dir = os.getcwd()
df = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
data = df.to_numpy()

# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
x = 1.0-data[:,0]/169.91 # i.e. Li concentration, divided by the theoretical capacity of LFP 
mu = -data[:,1]*96485 # because -mu_e- = OCV*F, -OCV*F = mu

# # AMYAO DEBUG
# plt.figure(figsize=(5,4))
# plt.plot(x, mu, 'k--')
# plt.show()
# exit()

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
G0_start = np.random.randint(-100,-50)*5000 # G0 is the pure substance gibbs free energy 
Omega0_start = np.random.randint(-100,100)*100
Omega1_start = np.random.randint(-100,100)*100
Omega2_start = np.random.randint(-100,100)*100
Omega3_start = np.random.randint(-100,100)*100
# Omega4_start = np.random.randint(-100,100)*100
G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
Omega0 = nn.Parameter( torch.from_numpy(np.array([Omega0_start],dtype="float32")) ) 
Omega1 = nn.Parameter( torch.from_numpy(np.array([Omega1_start],dtype="float32")) ) 
Omega2 = nn.Parameter( torch.from_numpy(np.array([Omega2_start],dtype="float32")) ) 
Omega3 = nn.Parameter( torch.from_numpy(np.array([Omega3_start],dtype="float32")) ) 
# Omega4 = nn.Parameter( torch.from_numpy(np.array([Omega4_start],dtype="float32")) ) 
# declare all params
params_list = [Omega0, Omega1, Omega2, Omega3, G0] 
# params_list = [Omega0, Omega1, Omega2, Omega3, Omega4, G0] 

# init optimizer
learning_rate = 1000.0
params_list_for_optimizer = [Omega0, Omega1, Omega2, Omega3, G0] 
# params_list_for_optimizer = [Omega0, Omega1, Omega2, Omega3, Omega4, G0] 
optimizer = optim.Adam(params_list_for_optimizer, lr=learning_rate)

# init loss weight
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
epoch = -1
while loss > 0.01 and epoch < 8000:
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
    sample = sampling(GibbsFE, params_list, T=300, sampling_id=1, ngrid=99)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is phase boundary predicted 
        common_tangent = CommonTangent(GibbsFE, params_list, T = 300) # init common tangent model
        phase_boundary_now = phase_boundarys_init[0] # we just take the first set of phase boundary as input. Hopefully after training it will become the only one
        phase_boundary_now = phase_boundary_now.requires_grad_()
        phase_boundary_fixed_point = common_tangent(phase_boundary_now) 
    else:
        # No boundary find.
        phase_boundary_fixed_point = torch.tensor([99999.9,-99999.9])
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
    output_txt = output_txt + "x_left %.4f "%(phase_boundary_fixed_point[0].item())
    output_txt = output_txt + "x_right %.4f "%(phase_boundary_fixed_point[1].item())
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
        SOC = 1.0 - x.clone().numpy()
        x_plot = 1.0 - x.clone().numpy()
        # plot figure
        plt.figure(figsize=(5,4))
        # plot the one before common tangent construction
        U_pred_before_ct = mu_pred/(-96485)
        plt.plot(SOC, U_pred_before_ct, 'k--', label="Prediction Before CT Construction")
        # plot the one after common tangent construction
        mu_pred_after_ct = []
        x_alpha = phase_boundary_fixed_point[0]
        x_beta = phase_boundary_fixed_point[1]
        if x_alpha < x_beta:
            ct_pred = (GibbsFE(x_alpha, params_list, T=300) - GibbsFE(x_beta, params_list, T=300))/(x_alpha - x_beta) 
        else:
            ct_pred = torch.tensor(torch.nan)
        if torch.isnan(ct_pred):
            mu_pred_after_ct = mu_pred * 1.0
        else:
            for i in range(0, len(x)):
                if x[i] < phase_boundary_fixed_point[0] or x[i] > phase_boundary_fixed_point[1]:
                    mu_pred_after_ct.append(mu_pred[i])
                else:
                    mu_pred_after_ct.append(ct_pred.clone().detach().numpy()[0])
            mu_pred_after_ct = np.array(mu_pred_after_ct)
        U_pred_after_ct = mu_pred_after_ct/(-96485)
        plt.plot(SOC, U_pred_after_ct, 'r-', label="Prediction After CT Construction")
        U_true_value = mu.numpy()/(-96485) # plot the true value
        plt.plot(SOC, U_true_value, 'b-', label="True OCV")
        plt.xlim([0,1])
        plt.ylim([2.5, 4.5])
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

# # # plot Gibbs free energy landscape
# mu_pred = []
# g_pred = []
# ngrid = 99
# x_pred = torch.from_numpy( np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid) )
# for i in range(0, len(x_pred)):
#     x_now = x_pred[i]
#     mu_now = mu[i]
#     x_now = x_now.requires_grad_()
#     g_now = GibbsFE(x_now, params_list, T=300)
#     g_pred.append(g_now.detach().numpy())
#     mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
#     mu_pred.append(mu_pred_now.detach().numpy())
# mu_pred = np.array(mu_pred)
# x_pred = 1.0 - x_pred.numpy()
# g_pred = np.array(g_pred)
# plt.figure(figsize=(5,4))
# plt.plot(x_pred, g_pred, label="pred")
# plt.xlim([0,1])
# plt.legend()
# plt.savefig("Final_g.png", bbox_inches='tight')
# plt.close()
exit()