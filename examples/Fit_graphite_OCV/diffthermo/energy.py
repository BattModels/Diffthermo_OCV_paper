import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from .solver import FixedPointOperation, FixedPointOperationForwardPass, newton_raphson

global  _eps
_eps = 1e-7



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
    G = x*G0 + (1-x)*0.0 + 8.314*T*(x*torch.log(x)+(1-x)*torch.log(1-x)) 
    for i in range(0, len(params_list)-1):
        G = G + x*(1-x)*(params_list[i]*(1-2*x)**i)
    return G



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
    # sort the init guess of convex hull
    left_sides = torch.zeros(len(miscibility_gap_x_left_and_right))
    for i in range(0, len(miscibility_gap_x_left_and_right)):
        left_sides[i] = miscibility_gap_x_left_and_right[i][0]
    _, index =  torch.sort(left_sides)
    miscibility_gap_x_left_and_right_sorted = []
    miscibility_gap_phase_left_and_right_sorted = []
    for _ in range(0, len(index)):
        miscibility_gap_x_left_and_right_sorted.append(miscibility_gap_x_left_and_right[_])
        miscibility_gap_phase_left_and_right_sorted.append(miscibility_gap_phase_left_and_right[_])
    return miscibility_gap_x_left_and_right_sorted, miscibility_gap_phase_left_and_right_sorted    



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

