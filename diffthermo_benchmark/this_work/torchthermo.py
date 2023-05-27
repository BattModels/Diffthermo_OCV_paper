import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim

global is_print , _eps
is_print = False # for debug only AMYAO DEBUG
_eps = 1e-7



class GibbsPureSubstance:
    """
    Expression for Gibbs Free Energy of a pure substance
    G0 = a + b*T + c*T*torch.log(T) + \sum_i d_i *T**i 
    where i>=2, a is actually a + H_SER
    the above expression is set according to Scientific Group Thermodata Europe database
    """
    def __init__(self, a, b, c, ds_list):
        self.a = a
        self.b = b
        self.c = c
        self.ds_list = ds_list
    def calculate(self, T):
        g0 = self.a + self.b*T + self.c*T*torch.log(T)
        for i in range(0, len(self.ds_list)):
            g0 = g0 + self.ds_list[i]**(i+2)
        return g0


def legendre(x, order):
    """
    Legendre Polynomials:
    Bonnet’s recursion formula tells us that   (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
    where P_{n}(x) is the n-th order polynomial
    The first several Legendre polynomials:
    P0 = 1
    P1 = x
    P2 = 1/2*(3*x**2-1)
    P3 = 1/2*(5*x**3 - 3*x)
    P4 = 1/8*(35*x**4 - 30*x**2 + 3)
    P5 = 1/8*(63*x**5 -70*x**3 + 15*x)
    """
    if order == 0:
        return x**order
    elif order == 1:
        return 1.0*x
    elif order == 2:
        return 1/2*(3*x**2-1)
    elif order == 3:
        return 1/2*(5*x**3 - 3*x)
    elif order == 4:
        return 1/8*(35*x**4 - 30*x**2 + 3)
    elif order == 5:
        return 1/8*(63*x**5 -70*x**3 + 15*x)
    else:
        print("NOT IMPLEMENTED LEGENDRE")
        exit()

class GibbsFE(nn.Module):
    """
    Expression for Gibbs Free Energy of a binary mixture
    """
    def __init__(self, rk_params_list, G0, G1, is_legendre = False):
        """
        Input:
        rk_params_list: the RK params, in the sequence of [Omega0, Omega1, ..., Omegan]
        G0: a function of T that defines the Gibbs free energy of pure substance 0
        G1: a function of T that defines the Gibbs free energy of pure substance 1
        is_legendre: whether to use Legendre polynomials when doing RK expansion
        """
        super(GibbsFE, self).__init__()
        self.rk_params_list = rk_params_list
        self.G0 = G0
        self.G1 = G1
        self.is_legendre = is_legendre
    def forward(self, x, T):
        """
        Input: 
        x: SOC
        T: temperature
        """
        x = torch.clamp(x, min=_eps, max=1.0-_eps)
        G = x*self.G0.calculate(T) + (1-x)*self.G1.calculate(T) + 8.314*T*(x*torch.log(x)+(1-x)*torch.log(1-x)) 
        if self.is_legendre == False:
            for i in range(0, len(self.rk_params_list)):
                G = G + x*(1-x)*(self.rk_params_list[i]*(1-2*x)**i)
        else:
            for i in range(0, len(self.rk_params_list)):
                G = G + x*(1-x)*(self.rk_params_list[i]*legendre(1-2*x, order = i))
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
        x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (2,)) # TODO shall we use pinv?
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp
        x_new[0] = torch.max(torch.tensor([0.0+_eps, x_new[0]]))
        x_new[1] = torch.min(torch.tensor([1.0-_eps, x_new[1]])) # +- _eps is for the sake of torch.log. We don't want log function to blow up at x=0!
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


def sampling(G_func, T, sampling_id, ngrid=99, requires_grad = False):
    """
    Sampling a Gibbs free energy function (G_func)
    sampling_id is for recognition, must be a interger
    T should be torch.tensor(T)
    """
    x = np.concatenate((np.array([1e-5]),np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),np.array([1-1e-5]))) 
    x = torch.from_numpy(x.astype("float32"))
    x = x.requires_grad_()
    sample = torch.tensor([[x[i], G_func(x[i], T), sampling_id] for i in range(0, len(x))])
    return sample


def convex_hull(samples_list, tolerance = 1e-5):
    """
    Convex Hull Algorithm that provides the initial guess for common tangent
    Need NOT to be differentiable
    returning the initial guess for common tangent & corresponding phase id
    Adapted from Pinwe's Jax-Thermo with some modifications
    Cite Pinwen's Jax-TherMo: https://github.com/PinwenGuan/JAX-TherMo
    Input:
    samples_list: a list of sample from different Gibbs free energy landscape, e.g. [sample1, sample2, sample3, ...], where sample1 is returned from the sampling function above (same for sample2, sample3 etc.)
    tolerance: the tolerance for detecting concave points
    """
    ngrid = samples_list[0].shape[0]-2
    # global minimization of gibbs free energy
    sample = torch.zeros((0,3))
    for i in range(0, samples_list[0].shape[0]):
        # select the one with the least 
        _gibbs_fe_now = []
        for _ in range(0, len(samples_list)):
            _gibbs_fe_now.append(samples_list[_][i,1])
        _gibbs_fe_now = np.array(_gibbs_fe_now)
        _min_index = np.argmin(_gibbs_fe_now)
        sample = torch.cat((sample, samples_list[_min_index][i:(i+1), :]), dim=0)
    # convex hull, starting from the furtest points at x=0 and 1 and find all pieces
    base = [[sample[0,:], sample[-1,:]]]
    current_base_length = len(base)
    new_base_length = 9999999
    base_working = base.copy()
    while new_base_length != current_base_length:
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
            _t = sample[:,0:2]-h[0:2]
            dists = torch.matmul(_t, _n).squeeze()
            # select those underneath the hyperplane
            outer = []
            for _ in range(0, sample.shape[0]):
                if dists[_]<-tolerance: # if we write dists[i] < 0, then probably the endpoints themselves (i.e. h and t) will be included
                    outer.append(sample[_,:]) 
            # if there are points underneath the hyperplane, select the farthest one
            if len(outer):
                pivot = sample[torch.argmin(dists)] # the furthest node below the hyperplane defined hy t[column]-h[column]
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
    def __init__(self, G_left, G_right, T):
        """
        The fixed point operation used in the backward pass of common tangent approach. 
        Write the forward(self, x) function in such weird way so that it is differentiable
        G_left: Gibbs free energy function for the left side (pass the predefined function here)
        G_right: Gibbs free energy function for the right side
        T: temperature
        """
        super(FixedPointOperation, self).__init__()
        self.G_left = G_left
        self.G_right = G_right
        self.T = T
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        x_alpha = x[0]
        x_beta = x[1]
        g_right = self.G_right(x_beta, self.T)  
        g_left = self.G_left(x_alpha, self.T)  
        mu_right = autograd.grad(outputs=g_right, inputs=x_beta, create_graph=True)[0]
        mu_left = autograd.grad(outputs=g_left, inputs=x_alpha, create_graph=True)[0]
        x_alpha_new = x_beta - (g_right - g_left)/mu_left 
        x_alpha_new = torch.clamp(x_alpha_new , min=0.0+1e-6, max=1.0-1e-6) # clamp
        x_alpha_new = x_alpha_new.reshape(1)
        x_beta_new = x_alpha + (g_right - g_left)/mu_right 
        x_beta_new = torch.clamp(x_beta_new , min=0.0+1e-6, max=1.0-1e-6) # clamp
        x_beta_new = x_beta_new.reshape(1)
        return torch.cat((x_alpha_new, x_beta_new))


# In case that the above implementation doesn't work
class FixedPointOperationForwardPass(nn.Module):
    def __init__(self, G_left, G_right, T):
        """
        The fixed point operation used in the forward pass of common tangent approach
        Here we don't use the above implementation (instead we use Pinwen's implementation) to guarantee that the solution converges to the correct places in forward pass
        G_left: Gibbs free energy function for the left side (pass the predefined function here)
        G_right: Gibbs free energy function for the right side
        T: temperature
        """
        super(FixedPointOperationForwardPass, self).__init__()
        self.G_left = G_left
        self.G_right = G_right
        self.T = T
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        # x_alpha_0 = (x[0]).reshape(1)
        # x_beta_0 = (x[1]).reshape(1)
        x_alpha_now = x[0]
        x_beta_now = x[1]
        x_alpha_now = x_alpha_now.reshape(1)
        x_beta_now = x_beta_now.reshape(1)
        g_right = self.G_right(x_beta_now, self.T)  
        g_left = self.G_left(x_alpha_now, self.T)  
        common_tangent = (g_left - g_right)/(x_alpha_now - x_beta_now)
        dcommon_tangent = 9999999.9
        n_iter_ct = 0
        while dcommon_tangent>1e-3 and n_iter_ct < 100:  
            """
            eq1 & eq2: we want dG/dx evaluated at x1 (and x2) to be the same as common_tangent, i.e. mu(x=x1 or x2) = common_tangent
            then applying Newton-Ralpson iteration to solve f(x) = mu(x) - common_tangent, where mu(x) = dG/dx
            Newton-Ralpson iteration: x1 = x0 - f(x0)/f'(x0)
            """  
            def eq1(x):
                y = self.G_left(x_alpha_now, self.T)   - common_tangent*x
                return y
            def eq2(x):
                y = self.G_right(x_beta_now, self.T)   - common_tangent*x
                return y
            # update x_alpha
            dx = torch.tensor(999999.0)
            n_iter_dxalpha = 0.0
            while torch.abs(dx) > 1e-5 and n_iter_dxalpha < 100:
                x_alpha_now = x_alpha_now.requires_grad_()
                value_now = eq1(x_alpha_now)
                f_now = autograd.grad(value_now, x_alpha_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_alpha_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_alpha_now = x_alpha_now + dx
                x_alpha_now = x_alpha_now.clone().detach()
                # clamp
                x_alpha_now = torch.max(torch.tensor([0.0+1e-6, x_alpha_now]))
                x_alpha_now = torch.min(torch.tensor([1.0-1e-6, x_alpha_now])) 
                x_alpha_now = x_alpha_now.reshape(1)
                n_iter_dxalpha = n_iter_dxalpha + 1
            # update x_beta
            dx = torch.tensor(999999.0)
            n_iter_dxbeta = 0.0
            while torch.abs(dx) > 1e-5 and n_iter_dxbeta < 100:
                x_beta_now = x_beta_now.requires_grad_()
                value_now = eq2(x_beta_now)
                f_now = autograd.grad(value_now, x_beta_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_beta_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_beta_now = x_beta_now + dx
                x_beta_now = x_beta_now.clone().detach()
                # clamp
                x_beta_now = torch.max(torch.tensor([0.0+1e-6, x_beta_now]))
                x_beta_now = torch.min(torch.tensor([1.0-1e-6, x_beta_now])) 
                x_beta_now = x_beta_now.reshape(1)
                n_iter_dxbeta = n_iter_dxbeta + 1
            # after getting new x1 and x2, calculates the new common tangent, the same process goes on until the solution is self-consistent
            common_tangent_new = (self.G_left(x_alpha_now, self.T) - self.G_right(x_beta_now, self.T) )/(x_alpha_now - x_beta_now)
            dcommon_tangent = torch.abs(common_tangent_new-common_tangent)
            common_tangent = common_tangent_new.clone().detach()
            n_iter_ct = n_iter_ct + 1
        return torch.cat((x_alpha_now, x_beta_now))


class CommonTangent(nn.Module):
    """
    Common Tangent Approach for phase equilibrium boundary calculation
    """
    def __init__(self, G_left, G_right, T):
        super(CommonTangent, self).__init__()
        self.f = FixedPointOperation(G_left, G_right, T) 
        self.f_forward = FixedPointOperationForwardPass(G_left, G_right, T) # in case that self.f doesn't work
        self.solver = newton_raphson
        self.f_thres = 1e-6
        self.T = T
    def forward(self, x, **kwargs):
        """
        x is the initial guess provided by convex hull
        """
        # Forward pass
        x_star = self.solver(self.f, x, threshold=self.f_thres) # use newton-ralphson to get the fixed point
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


def hessian_loss(G_func, T, threshold = 1e-4):
    """
    For computing the loss function if no miscibility gap is predicted in the current Gibbs free energy landscape.
    Please note that the driving force loss is only for the case that no phase-coexistence between two phases are predicted.
    Input:
    G_func: the Gibbs free energy function, should be instatiated from class GibbsFE
    Omega0 & Omega1: params in G_function, should be nn.Parameter
    T: temperature
    """
    ngrid = 99
    x = np.concatenate((np.array([1e-5]),np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),np.array([1-1e-5]))) 
    x = torch.from_numpy(x.astype("float32"))
    x = x.requires_grad_()
    def g(x_):
        return G_func(x_, T)
    hessian_sample = torch.tensor([[autograd.functional.hessian(g, x[i])] for i in range(0, len(x))])
    x_now = x[torch.argmin(hessian_sample)].detach()
    error = 9999999.9
    # print("Initial guess hessian = %.4f at x = %.4f" %(hessian_sample[torch.argmin(hessian_sample)].item(), x[torch.argmin(hessian_sample)].item()))
    def calc_hessian(x_):
        return autograd.functional.hessian(g, x_, create_graph=True)
    while error > threshold:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        f_now = calc_hessian(x_now)
        f_now_prime = autograd.grad(f_now, x_now)[0]
        x_new = x_now - f_now/f_now_prime # newton-raphson
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp
        x_new = torch.max(torch.tensor([0.0+1e-6, x_new]))
        x_new = torch.min(torch.tensor([1.0-1e-6, x_new])) 
        # sometimes, the newton-raphson explodes at x=1 or 0 without giving the minimum value. 
        # In these cases, we just return the approximate guess giving by the torch.argmin of hessian samples
        f_new = calc_hessian(x_new)
        if f_new < f_now:
            x_now = x_now.clone().detach() # detach for memory saving
            # calculate error
            error = torch.abs(x_new - x_now)
            # step forward
            x_now = x_new.clone()
        else:
            x_new = x_now.clone()
            error = 0.0
    # find the x where hessian(x) is minimized
    x_min = x_now.requires_grad_()
    hessian_min = autograd.functional.hessian(g, x_min, create_graph=True)
    loss = torch.nn.functional.relu((hessian_min/(8.314*T))) # relu is because we don't want negative hessian to contribute to loss
    return loss**2


def driving_force_loss(G_func1, G_func2, T, threshold = 1e-4):
    """
    For computing the loss function if no phase-coexistence between two phases are predicted.
    Rewrite according to Pinwen's Jax-Thermo [Cite Pinwen's Jax-Thermo manuscript!]
    Input:
    G_func1: the Gibbs free energy function 1, should be instatiated from class GibbsFE (it is supposed to be the ground state)
    G_func1: the Gibbs free energy function 2, should be instatiated from class GibbsFE 
    T: temperature
    """
    # we first define a function that calculates the distance of metastable phase to the stable phase at x 
    # and then minimize this distance for all x
    def distance_of_plane_at_x(x):
        # find the minimum gibbs function 
        if G_func2(x, T) > G_func1(x, T):
            GibbsFunction1_here = G_func2
            GibbsFunction2_here = G_func1
        else:
            GibbsFunction1_here = G_func1
            GibbsFunction2_here = G_func2
        # define the hyperplane
        def g(x_here):
            return GibbsFunction1_here(x_here, T)
        x = x.requires_grad_()
        g_value = g(x)
        k = autograd.grad(g_value, x, create_graph=True)[0]
        ngrid = 99
        x_sample = np.concatenate((np.array([1e-5]),np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),np.array([1-1e-5]))) 
        x_sample = torch.from_numpy(x_sample.astype("float32"))
        distance = torch.tensor([torch.abs(k*(x_sample_now-x)-(GibbsFunction2_here(x_sample_now, T) - GibbsFunction1_here(x, T)))/(1+k**2) for x_sample_now in x_sample])
        x_metastable = x_sample[torch.argmin(distance)]
        # now we want to find the x* such that d GibbsFunction2 / dx (x = x*)  = k, i.e. the plane on the metastable phase which is parallel to the plane on the stable Gibbs free energy landscape
        def _func(x_here):
            x_here = x_here.requires_grad_()
            g_ = g(x_here)
            k = autograd.grad(g_, x_here, create_graph=True)[0]
            return GibbsFunction2_here(x_here, T) - k * x_here
        error = 9999999.9
        x_now = x_metastable.clone()
        while error > threshold:
            x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
            f_now = _func(x_now)
            f_now_prime = autograd.grad(f_now, x_now)[0]
            x_new = x_now - f_now/f_now_prime # newton-raphson
            # detach for memory saving
            x_new = x_new.clone().detach() # detach for memory saving
            # clamp
            x_new = torch.max(torch.tensor([0.0+1e-6, x_new]))
            x_new = torch.min(torch.tensor([1.0-1e-6, x_new])) 
            # sometimes, the newton-raphson explodes at x=1 or 0 without giving the minimum value. 
            # In these cases, we just return the approximate guess giving by the torch.argmin of hessian samples
            f_new = _func(x_new)
            if f_new < f_now:
                x_now = x_now.clone().detach() # detach for memory saving
                # calculate error
                error = torch.abs(x_new - x_now)
                # step forward
                x_now = x_new.clone()
            else:
                x_now = x_now.clone().detach() # detach for memory saving
                x_new = x_now.clone()
                error = 0.0
        driving_force_here = (GibbsFunction2_here(x_now, T) - GibbsFunction1_here(x_now, T) - k*(x_now-x))/(8.314*T)
        return driving_force_here
    # now minimize this distance for all x
    ngrid = 99
    x_sample = np.concatenate((np.array([1e-5]),np.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),np.array([1-1e-5]))) 
    x_sample = torch.from_numpy(x_sample.astype("float32"))
    driving_force_sample=torch.tensor([distance_of_plane_at_x(x_here) for x_here in x_sample])
    x_0 = x_sample[torch.argmin(driving_force_sample)] # initial guess for Newton-Raphson
    # now we want to find the x* such that distance_of_plane_at_x(x*) is minimized, i.e. d distance_of_plane_at_x/dx (x = x*) = 0
    # therefore we use Newton-Ralphson method to solve the equation that d distance_of_plane_at_x/dx (x = x*) = 0
    error = 9999999.9
    x_now = x_0.clone()
    while error > threshold:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        driving_force_now = distance_of_plane_at_x(x_now)
        f_now = autograd.grad(driving_force_now, x_now, create_graph=True)[0]
        f_now_prime = autograd.grad(f_now, x_now, create_graph=True)[0]
        x_new = x_now - f_now/f_now_prime # newton-raphson
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp
        x_new = torch.max(torch.tensor([0.0+1e-6, x_new]))
        x_new = torch.min(torch.tensor([1.0-1e-6, x_new])) 
        # sometimes, the newton-raphson explodes at x=1 or 0 without giving the minimum value. 
        # In these cases, we just return the approximate guess giving by the torch.argmin of hessian samples
        driving_force_new = distance_of_plane_at_x(x_new)      
        if driving_force_new < driving_force_now:
            x_now = x_now.clone().detach() # detach for memory saving
            # calculate error
            error = torch.abs(x_new - x_now)
            # step forward
            x_now = x_new.clone()
        else:
            x_now = x_now.clone().detach() # detach for memory saving
            x_new = x_now.clone()
            error = 0.0
    driving_force_min = distance_of_plane_at_x(x_now)
    loss = driving_force_min**2
    return loss





# set the ground truth. Same as Pinwen's paper at 1000K
# cit, sampling_id = 1
Omega0_1_true = 20134.5
Omega1_1_true = -5525.5
# scs, sampling_id = 2
Omega0_2_true = 18313.5
Omega1_2_true = -9899.5
print("True Value:                                                              Omega0_cit %.4f Omega1_cit %.4f  Omega0_scs %.4f Omega1_scs %.4f" %(Omega0_1_true, Omega1_1_true, Omega0_2_true, Omega1_2_true))

G0_zero_const = GibbsPureSubstance(0.0,0.0,0.0,[])
G_cit_true = GibbsFE([Omega0_1_true,Omega1_1_true], G0_zero_const, G0_zero_const) #     G = x*0 + (1-x)*0 + 8.314*T*(x*torch.log(x)+(1-x)*torch.log(1-x)) + x*(1-x)*(20134.5 - 5525.5*(1-2*x))
G0_1_scs = GibbsPureSubstance(18753.0, -8.372, 0.0, [])
G0_2_scs = GibbsPureSubstance(13054.0, -9.623, 0.0, [])
G_scs_true = GibbsFE([Omega0_2_true,Omega1_2_true], G0_1_scs, G0_2_scs) # true value: G = x*(18753-8.372*T) + (1-x)*(13054-9.623*T) + 8.314*T*(x*torch.log(x)+(1-x)*torch.log(1-x)) + x*(1-x)*(18313.5 -9899.5*(1-2*x))


# construct the true boundary
Ts = [1000,1200,1400,1600,1800,2000,2200]
# Ts = [1000]
phase_boundarys = []
corresponding_phase_ids = []
for T in Ts:
    T = torch.tensor([T])
    sample1 = sampling(G_cit_true, T=T, sampling_id=1, ngrid=99)
    sample2 = sampling(G_scs_true, T=T, sampling_id=2, ngrid=99)
    phase_boundarys_at_T_init, corresponding_phase_ids_at_T_init = convex_hull([sample1, sample2]) # there might be multiple sets of boundaries in phase_boundary_0 and corresponding_phase_id lists!
    phase_boundarys_at_T = []
    corresponding_phase_ids_at_T = []
    for _ in range(0, len(phase_boundarys_at_T_init)):
        phase_boundary_now = phase_boundarys_at_T_init[_]
        corresponding_phase_id_now = corresponding_phase_ids_at_T_init[_]
        if torch.all(corresponding_phase_id_now==torch.tensor([1,1])):
            common_tangent = CommonTangent(G_left=G_cit_true, G_right=G_cit_true, T = T) # init
        elif torch.all(corresponding_phase_id_now==torch.tensor([1,2])):
            common_tangent = CommonTangent(G_left=G_cit_true, G_right=G_scs_true, T = T) # init
        elif torch.all(corresponding_phase_id_now==torch.tensor([2,1])):
            common_tangent = CommonTangent(G_left=G_scs_true, G_right=G_cit_true, T = T) # init
        elif torch.all(corresponding_phase_id_now==torch.tensor([2,2])):
            common_tangent = CommonTangent(G_left=G_scs_true, G_right=G_scs_true, T = T) # init
        phase_boundary_now = phase_boundary_now.requires_grad_()
        phase_boundary_fixed_point = common_tangent(phase_boundary_now)
        phase_boundarys_at_T.append(phase_boundary_fixed_point.detach())
        corresponding_phase_ids_at_T.append(corresponding_phase_id_now)
    phase_boundarys.append(phase_boundarys_at_T)
    corresponding_phase_ids.append(corresponding_phase_ids_at_T)

# # true values:
# print(phase_boundarys)            # [[tensor([0.2381, 0.9382])], [tensor([0.3768, 0.8660])], [tensor([0.1120, 0.1636])], [tensor([0.2641, 0.6673])], [tensor([0.3723, 0.8099])], [tensor([0.5523, 0.8654])], [tensor([0.9348, 0.9590])]]
# print(corresponding_phase_ids)    # [[tensor([1., 1.])],         [tensor([1., 1.])],         [tensor([2., 1.])],         [tensor([2., 1.])],         [tensor([2., 1.])],         [tensor([2., 1.])],         [tensor([2., 1.])]]
# exit()

# init log
with open("log",'w') as fin:
    fin.write("")

# init params that wait for training 
Omega0_cit_start = 6634.5 # true value 20134.5, should start from 6634.4 according to Pinwen
Omega1_cit_start = -1025.5 # true value -5525.5, should start from -1025.5 according to Pinwen
Omega0_cit = nn.Parameter( torch.from_numpy(np.array([Omega0_cit_start],dtype="float32")) ) 
Omega1_cit = nn.Parameter( torch.from_numpy(np.array([Omega1_cit_start],dtype="float32")) ) 
Omega0_scs_start = 22813.5 # true value 18313.5, should start from 22813.5 according to Pinwen
Omega1_scs_start = -5399.5 # true value -9899.5, should start from -5399.5 according to Pinwen
Omega0_scs = nn.Parameter( torch.from_numpy(np.array([Omega0_scs_start],dtype="float32")) ) 
Omega1_scs = nn.Parameter( torch.from_numpy(np.array([Omega1_scs_start],dtype="float32")) ) 

# init optimizer
learning_rate = 1000.0
params_list = [Omega0_cit, Omega1_cit, Omega0_scs, Omega1_scs]
optimizer = optim.Adam(params_list, lr=learning_rate)

# init Gibbs free energy models
G_cit = GibbsFE([Omega0_cit, Omega1_cit], G0_zero_const, G0_zero_const)  
G_scs = GibbsFE([Omega0_scs, Omega1_scs], G0_1_scs, G0_2_scs) 

# train
Omega0_cit_list = []
Omega1_cit_list = []
Omega0_scs_list = []
Omega1_scs_list = []
epoch_list = []
# total_epochs = 1000
# for epoch in range(0, total_epochs):
loss = 9999.9 # init total loss
epoch = -1
while loss > 0.0001:
    # clean grad info
    optimizer.zero_grad()
    # use current params to calculate predicted phase boundary
    phase_boundarys_pred = []
    corresponding_phase_ids_pred = []
    epoch = epoch + 1
    loss = 0.0 # init total loss
    loss_pb = 0.0 # init loss func for phase boundary
    n_pb = 0 # count how many phase boundary contributions are there
    loss_df = 0.0 # init loss func for driving force
    n_df = 0 # count how many driving force contributions are there
    loss_hessian = 0.0 # init loss func for hessian 
    n_hessian = 0 # count how many hessian contributions are there
    for i in range(0, len(Ts)):
        # sample
        T = Ts[i]
        T = torch.tensor([T])
        sample1 = sampling(G_cit, T=T, sampling_id=1, ngrid=99)
        sample2 = sampling(G_scs, T=T, sampling_id=2, ngrid=99)
        # initialze model
        phase_boundarys_at_T_init, corresponding_phase_ids_at_T_init = convex_hull([sample1, sample2]) # there might be multiple sets of boundaries in phase_boundary_0 and corresponding_phase_id lists!
        phase_boundarys_at_T = []
        corresponding_phase_ids_at_T = []
        # If there is phase boundary predicted
        # Here we know there's only one set of phase coexistence / separation at all Temperature, so we can write like this
        for _ in range(0, len(phase_boundarys_at_T_init)):  
            phase_boundary_now = phase_boundarys_at_T_init[_]
            corresponding_phase_id_now = corresponding_phase_ids_at_T_init[_]
            # here we know there's only one set of phase coexistence / separation at all Temperature, so we can write like this
            if torch.all(corresponding_phase_id_now == corresponding_phase_ids[i][0])==True:          # find the one that matches phase ids
                if torch.all(corresponding_phase_id_now==torch.tensor([1,1])):
                    common_tangent = CommonTangent(G_left=G_cit, G_right=G_cit, T = T) # init
                elif torch.all(corresponding_phase_id_now==torch.tensor([1,2])):
                    common_tangent = CommonTangent(G_left=G_cit, G_right=G_scs, T = T) # init
                elif torch.all(corresponding_phase_id_now==torch.tensor([2,1])):
                    common_tangent = CommonTangent(G_left=G_scs, G_right=G_cit, T = T) # init
                elif torch.all(corresponding_phase_id_now==torch.tensor([2,2])):
                    common_tangent = CommonTangent(G_left=G_scs, G_right=G_scs, T = T) # init
                phase_boundary_now = phase_boundary_now.requires_grad_()
                phase_boundary_fixed_point = common_tangent(phase_boundary_now) 
                loss_pb_now = torch.sum((phase_boundary_fixed_point - phase_boundarys[i][0])**2)
                # here we know there's only one set of phase coexistence / separation at all Temperature, so we can write like this
                if torch.isnan(loss_pb_now) == False:
                    loss_pb = loss_pb + loss_pb_now
                    n_pb = n_pb + 1
                    phase_boundarys_at_T.append(phase_boundary_fixed_point) # just for reference
                    corresponding_phase_ids_at_T.append(corresponding_phase_id_now) # just for reference
                else:
                    print("loss_pb nan at T=%d. " %(T), phase_boundary_now, "     ", phase_boundary_fixed_point, phase_boundarys[i][0], corresponding_phase_id_now, corresponding_phase_ids[i][0])
        if corresponding_phase_ids_at_T_init == []:
            # No boundary find. Use either driving force mode or phase separation (hessian) mode
            if corresponding_phase_ids[i][0][0] == corresponding_phase_ids[i][0][1]: 
                # miscibility gap should exist. Use Phase separation mode (hessian loss)
                if torch.all(corresponding_phase_ids[i][0] == torch.tensor([1,1])):
                    loss_hessian_now = hessian_loss(G_cit, T = T)
                elif torch.all(corresponding_phase_ids[i][0] == torch.tensor([2,2])):
                    loss_hessian_now = hessian_loss(G_scs, T = T)
                if torch.isnan(loss_hessian_now) == False:
                    loss_hessian = loss_hessian + loss_hessian_now
                    n_hessian = n_hessian + 1
                else:
                    print("loss_hessian nan at T=%d . " %(T), corresponding_phase_ids[i][0])
            else:
                # phase coexistence should exist. Use driving force mode!
                if torch.all(corresponding_phase_ids[i][0] == torch.tensor([1,2])):
                    loss_df_now = driving_force_loss(G_cit, G_scs, T = T)
                elif torch.all(corresponding_phase_ids[i][0] == torch.tensor([2,1])):
                    loss_df_now = driving_force_loss(G_scs, G_cit, T = T)
                if torch.isnan(loss_df_now) == False:
                    loss_df = loss_df + loss_df_now
                    n_df = n_df + 1
                else:
                    print("loss_df nan at T=%d . " %(T), corresponding_phase_ids[i][0])
    loss = 100.0*loss_pb/(n_pb+1e-8) + loss_hessian/(n_hessian+1e-8) + loss_df/(n_df+1e-8)
    loss.backward()
    optimizer.step()
    # record
    Omega0_cit_list.append(Omega0_cit.item()/1000.0)
    Omega1_cit_list.append(Omega1_cit.item()/1000.0)
    Omega0_scs_list.append(Omega0_scs.item()/1000.0)
    Omega1_scs_list.append(Omega1_scs.item()/1000.0)
    epoch_list.append(epoch)
    print("Epoch %3d  Loss %.4f     Pb %.4f %d Hessian %.4f %d Df %.4f %d    Omega0_cit %.4f Omega1_cit %.4f  Omega0_scs %.4f Omega1_scs %.4f" %(epoch, loss, 100.0*loss_pb/(n_pb+1e-8), n_pb, loss_hessian/(n_hessian+1e-8), n_hessian, loss_df/(n_df+1e-8), n_df, Omega0_cit, Omega1_cit, Omega0_scs, Omega1_scs))
    with open("log",'a') as fin:
        fin.write("Epoch %3d  Loss %.4f     Pb %.4f %d Hessian %.4f %d Df %.4f %d    Omega0_cit %.4f Omega1_cit %.4f  Omega0_scs %.4f Omega1_scs %.4f\n" %(epoch, loss, 100.0*loss_pb/(n_pb+1e-8), n_pb, loss_hessian/(n_hessian+1e-8), n_hessian, loss_df/(n_df+1e-8), n_df, Omega0_cit, Omega1_cit, Omega0_scs, Omega1_scs))

print("Training Complete.\n")

total_epochs = len(epoch_list)
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
plt.plot(epoch_list, Omega0_cit_list, 'r-', label="Omega0_cit")
Omega0_true_list = [Omega0_1_true/1000.0]*total_epochs
plt.plot(epoch_list, Omega0_true_list, 'k--', label="True Value of Omega0_cit")
plt.xlim([0,total_epochs])
plt.xlabel("Epoch")
plt.ylabel("Param")
plt.legend()
plt.savefig("Omega0_cit.png",bbox_inches='tight')
plt.close()

plt.figure(figsize=(5,4))
plt.plot(epoch_list, Omega1_cit_list, 'b-', label="Omega1_cit")
Omega1_true_list = [Omega1_1_true/1000.0]*total_epochs
plt.plot(epoch_list, Omega1_true_list, 'k--', label="True Value of Omega1_cit")
plt.xlim([0,total_epochs])
plt.xlabel("Epoch")
plt.ylabel("Param")
plt.legend()
plt.savefig("Omega1_cit.png",bbox_inches='tight')
plt.close()

plt.figure(figsize=(5,4))
plt.plot(epoch_list, Omega0_scs_list, 'r-', label="Omega0_scs")
Omega0_true_list = [Omega0_2_true/1000.0]*total_epochs
plt.plot(epoch_list, Omega0_true_list, 'k--', label="True Value of Omega0_scs")
plt.xlim([0,total_epochs])
plt.xlabel("Epoch")
plt.ylabel("Param")
plt.legend()
plt.savefig("Omega0_scs.png",bbox_inches='tight')
plt.close()

plt.figure(figsize=(5,4))
plt.plot(epoch_list, Omega1_scs_list, 'b-', label="Omega1_scs")
Omega1_true_list = [Omega1_2_true/1000.0]*total_epochs
plt.plot(epoch_list, Omega1_true_list, 'k--', label="True Value of Omega1_scs")
plt.xlim([0,total_epochs])
plt.xlabel("Epoch")
plt.ylabel("Param")
plt.legend()
plt.savefig("Omega1_scs.png",bbox_inches='tight')
plt.close()



