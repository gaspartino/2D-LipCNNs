import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.nn.utils.parametrizations as param
from layer import *

def generic_power_method(affine_fun, input_size, eps=1e-8,
                         max_iter=500, use_cuda=False):
    """ Return the highest singular value of the linear part of
    `affine_fun` and it's associated left / right singular vectors.

    INPUT:
        * `affine_fun`: an affine function
        * `input_size`: size of the input
        * `eps`: stop condition for power iteration
        * `max_iter`: maximum number of iterations
        * `use_cuda`: set to True if CUDA is present

    OUTPUT:
        * `eigenvalue`: maximum singular value of `affine_fun`
        * `v`: the associated left singular vector
        * `u`: the associated right singular vector

    NOTE:
        This algorithm is not deterministic, depending of the random
        initialisation, the returned eigenvectors are defined up to the sign.

        If affine_fun is a PyTorch model, beware of setting to `False` all
        parameters.requires_grad.

    TEST::
        >>> conv = nn.Conv2d(3, 8, 5)
        >>> for p in conv.parameters(): p.requires_grad = False
        >>> s, u, v = generic_power_method(conv, [1, 3, 28, 28])
        >>> bias = conv(torch.zeros([1, 3, 28, 28]))
        >>> linear_fun = lambda x: conv(x) - bias
        >>> torch.norm(linear_fun(v) - s * u) # should be very small

    TODO: more tests with CUDA
    """
    zeros = torch.zeros(input_size)
    if use_cuda:
        zeros = zeros.cuda()
    bias = affine_fun(Variable(zeros))
    linear_fun = lambda x: affine_fun(x) - bias

    def norm(x, p=2):
        """ Norm for each batch
        """
        norms = Variable(torch.zeros(x.shape[0]))
        if use_cuda:
            norms = norms.cuda()
        for i in range(x.shape[0]):
            norms[i] = x[i].norm(p=p)
        return norms

    # Initialise with random values
    v = torch.randn(input_size)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
    if use_cuda:
        v = v.cuda()

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(linear_fun, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    # Compute Rayleigh product to get eigenvalue
    u = linear_fun(Variable(v))  # unormalized left singular vector
    eigenvalue = norm(u)
    u = u.div(eigenvalue)
    return eigenvalue.item()

def _norm_gradient_sq(linear_fun, v):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fun(v))**2
    loss.backward(retain_graph=True)
    return v.grad.data

def _power_method_matrix(matrix, eps=1e-6, max_iter=300, use_cuda=False):
    """ Return square of maximal singular value of `matrix`
    """
    M = matrix.t() @ matrix
    v = torch.randn(M.shape[1], 1)
    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = M @ v
        v = v / torch.norm(v)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    # Compute Rayleigh product to get eivenvalue
    eigenvalue = torch.norm(matrix @ v)
    return eigenvalue, v

def lipschitz_upper_bound(model):
    lipschitz_constant = 1.0

    input_size = [1, 1, 32, 32]
    lipschitz_constant *= generic_power_method(model.model[0], input_size)
    input_size = [1, 16, 16, 16]
    lipschitz_constant *= generic_power_method(model.model[2], input_size)
    input_size = [1, 100, 32*8*8]
    lipschitz_constant *= generic_power_method(model.model[5], input_size)
    input_size = [1, 10, 100]
    lipschitz_constant *= generic_power_method(model.model[7], input_size)
    # ReLU layers do not affect the Lipschitz constant
  
    #for layer in model.model:
    #    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #        input_size = x.size() if isinstance(x, torch.Tensor) else None
    #        lipschitz_constant *= generic_power_method(layer, input_size)
        
    
    return lipschitz_constant