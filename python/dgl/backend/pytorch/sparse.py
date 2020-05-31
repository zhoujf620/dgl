import torch as th
from .tensor import zerocopy_to_dgl_ndarray as to_dgl_nd
from .tensor import zerocopy_from_dgl_ndarray as from_dgl_nd
from ... import kernel2 as K

def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension

    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = th.nonzero(th.tensor(grad_shape) - th.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])

class UMulESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_e_sum('mul', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = u_mul_e_sum(g.reverse(), dZ, Y)
            dX = _reduce_grad(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = u_mul_v(g, X, Y)
            dY = _reduce_grad(dY, Y.shape)
        return None, dX, dY

class UAddESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_e_sum('add', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = copy_u_sum(g.reverse(), dZ)
            dX = _reduce_grad(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = copy_u(g.reverse(), dZ)
            dY = _reduce_grad(dY, Y.shape)
        return None, dX, dY

class UMulEMax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_max('mul', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z),
                     to_dgl_nd(argX),
                     to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y, argX, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            argX = argX.long()
            argY = argY.long()
        assert False
        if ctx.needs_input_grad[1]:
            dX = th.zeros_like(X)
            deltaX = _reduce_grad(Y[argY] * dZ, X.shape)
            view_shape = (argX.shape[0],) + (1,) * (deltaX.ndim - 1)
            idx = argX.view(*view_shape).expand(*deltaX.shape)
            dX.scatter_add_(0, idx, deltaX)
        if ctx.needs_input_grad[2]:
            dY = th.zeros_like(Y)
            dY[argY] = _reduce_grad(X[argX] * dZ, Y.shape)
        return None, dX, dY

class UMulEMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_min('mul', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z),
                     to_dgl_nd(argX),
                     to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y, argX, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return UMulEMax.backward(ctx, dZ)

class UAddEMax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_max('add', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z),
                     to_dgl_nd(argX),
                     to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y, argX, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = th.zeros_like(X)
            deltaX = _reduce_grad(dZ, X.shape)
            view_shape = (argX.shape[0],) + (1,) * (deltaX.ndim - 1)
            idx = argX.view(*view_shape).expand(*deltaX.shape).long()
            dX.scatter_add_(0, idx, deltaX)
        if ctx.needs_input_grad[2]:
            dY = th.zeros_like(Y)
            dY[argY.long()] = _reduce_grad(dZ, Y.shape)
        return None, dX, dY

class UAddEMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_min('add', g,
                     to_dgl_nd(X),
                     to_dgl_nd(Y),
                     to_dgl_nd(Z),
                     to_dgl_nd(argX),
                     to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y, argX, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return UAddEMax.backward(ctx, dZ)

class CopyU(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        Z = th.zeros((g.number_of_edges(0), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        K.copy_u(g, to_dgl_nd(X), to_dgl_nd(Z))
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dX = None
        if ctx.needs_input_grad[1]:
            dX = copy_e_sum(g.reverse(), dZ)
        return None, dX

class CopyESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, Y):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + Y.shape[1:], device=Y.device, dtype=Y.dtype)
        K.copy_e_sum(g, to_dgl_nd(Y), to_dgl_nd(Z))
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dY = None
        if ctx.needs_input_grad[1]:
            dY = copy_u(g, dZ)
        return None, dY

class CopyEMax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, Y):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + Y.shape[1:], device=Y.device, dtype=Y.dtype)
        argY = th.zeros(Z.shape, device=Y.device, dtype=getattr(th, g.dtype))
        K.copy_e_max(g, to_dgl_nd(Y), to_dgl_nd(Z), to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(Y, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        Y, argY = ctx.saved_tensors
        dY = None
        if ctx.needs_input_grad[1]:
            dY = th.zeros_like(Y)
            dY[argY.long()] = dZ
        return None, dY

class CopyEMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, Y):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + Y.shape[1:], device=Y.device, dtype=Y.dtype)
        argY = th.zeros(Z.shape, device=Y.device, dtype=getattr(th, g.dtype))
        K.copy_e_min(g, to_dgl_nd(Y), to_dgl_nd(Z), to_dgl_nd(argY))
        ctx.backward_cache = g
        ctx.save_for_backward(Y, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return CopyEMax.backward(ctx, dZ)

class CopyUSum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        K.copy_u_sum(g, to_dgl_nd(X), to_dgl_nd(Z))
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dX = None
        if ctx.needs_input_grad[1]:
            dX = copy_u_sum(g.reverse(), dZ)
        return None, dX

class CopyUMax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.copy_u_max(g, to_dgl_nd(X), to_dgl_nd(Z), to_dgl_nd(argX))
        ctx.backward_cache = g
        ctx.save_for_backward(X, argX)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, argX = ctx.saved_tensors
        dX = None
        if ctx.needs_input_grad[1]:
            dX = th.zeros_like(X)
            view_shape = (argX.shape[0],) + (1,) * (dZ.ndim - 1)
            idx = argX.view(*view_shape).expand(*dZ.shape).long()
            dX.scatter_add_(0, idx, dZ)
        return None, dX

class CopyUMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        dtid = g.number_of_ntypes() - 1
        Z = th.zeros((g.number_of_nodes(dtid), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        argX = th.zeros(Z.shape, device=X.device, dtype=getattr(th, g.dtype))
        K.copy_u_min(g, to_dgl_nd(X), to_dgl_nd(Z), to_dgl_nd(argX))
        ctx.backward_cache = g
        ctx.save_for_backward(X, argX)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return CopyUMax.backward(ctx, dZ)

class UMulV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_edges(0),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_v('mul', g, to_dgl_nd(X), to_dgl_nd(Y), to_dgl_nd(Z))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = u_mul_e_sum(g.reverse(), Y, dZ)
            dX = _reduce_grad(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = u_mul_e_sum(g, X, dZ)
            dY = _reduce_grad(dY, Y.shape)
        return None, dX, dY

class UAddV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_edges(0),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_v('add', g, to_dgl_nd(X), to_dgl_nd(Y), to_dgl_nd(Z))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = copy_e_sum(g.reverse(), dZ)
            dX = _reduce_grad(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = copy_e_sum(g, dZ)
            dY = _reduce_grad(dY, Y.shape)
        return None, dX, dY

class UDotV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:-1], Y.shape[1:-1])
        Z = th.zeros((g.number_of_edges(0),) + bshape + (1,), device=X.device, dtype=X.dtype)
        K.u_op_v('dot', g, to_dgl_nd(X), to_dgl_nd(Y), to_dgl_nd(Z))
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return UMulV.backward(ctx, dZ)

copy_e_sum = CopyESum.apply
copy_e_max = CopyEMax.apply
copy_e_min = CopyEMin.apply
copy_u_sum = CopyUSum.apply
copy_u_max = CopyUMax.apply
copy_u_min = CopyUMin.apply
copy_u = CopyU.apply
copy_v = lambda g, X : copy_u(g.reverse(), X)

u_add_e_sum = UAddESum.apply
u_mul_e_sum = UMulESum.apply
u_sub_e_sum = lambda g, X, Y : u_add_e_sum(g, X, -Y)
u_div_e_sum = lambda g, X, Y : u_mul_e_sum(g, X, 1. / Y)

e_add_u_sum = lambda g, X, Y : u_add_e_sum(g, Y, X)
e_mul_u_sum = lambda g, X, Y : u_mul_e_sum(g, Y, X)
e_sub_u_sum = lambda g, X, Y : e_add_u_sum(g, X, -Y)
e_div_u_sum = lambda g, X, Y : e_mul_u_sum(g, X, 1. / Y)

e_add_v_sum = lambda g, X, Y : u_add_e_sum(g.reverse(), Y, X)
e_mul_v_sum = lambda g, X, Y : u_mul_e_sum(g.reverse(), Y, X)
e_sub_v_sum = lambda g, X, Y : e_add_v_sum(g, X, -Y)
e_div_v_sum = lambda g, X, Y : e_mul_v_sum(g, X, 1. / Y)

v_add_e_sum = lambda g, X, Y : e_add_v_sum(g, Y, X)
v_mul_e_sum = lambda g, X, Y : e_mul_v_sum(g, Y, X)
v_sub_e_sum = lambda g, X, Y : v_add_e_sum(g, X, -Y)
v_div_e_sum = lambda g, X, Y : v_mul_e_sum(g, X, 1. / Y)

u_add_e_max = UAddEMax.apply
u_mul_e_max = UMulEMax.apply
u_sub_e_max = lambda g, X, Y : u_add_e_max(g, X, -Y)
u_div_e_max = lambda g, X, Y : u_mul_e_max(g, X, 1. / Y)

e_add_u_max = lambda g, X, Y : u_add_e_max(g, Y, X)
e_mul_u_max = lambda g, X, Y : u_mul_e_max(g, Y, X)
e_sub_u_max = lambda g, X, Y : e_add_u_max(g, X, -Y)
e_div_u_max = lambda g, X, Y : e_mul_u_max(g, X, 1. / Y)

e_add_v_max = lambda g, X, Y : u_add_e_max(g.reverse(), Y, X)
e_mul_v_max = lambda g, X, Y : u_mul_e_max(g.reverse(), Y, X)
e_sub_v_max = lambda g, X, Y : e_add_v_max(g, X, -Y)
e_div_v_max = lambda g, X, Y : e_mul_v_max(g, X, 1. / Y)

v_add_e_max = lambda g, X, Y : e_add_v_max(g, Y, X)
v_mul_e_max = lambda g, X, Y : e_mul_v_max(g, Y, X)
v_sub_e_max = lambda g, X, Y : v_add_e_max(g, X, -Y)
v_div_e_max = lambda g, X, Y : v_mul_e_max(g, X, 1. / Y)

u_add_e_min = UAddEMin.apply
u_mul_e_min = UMulEMin.apply
u_sub_e_min = lambda g, X, Y : u_add_e_min(g, X, -Y)
u_div_e_min = lambda g, X, Y : u_mul_e_min(g, X, 1. / Y)

e_add_u_min = lambda g, X, Y : u_add_e_min(g, Y, X)
e_mul_u_min = lambda g, X, Y : u_mul_e_min(g, Y, X)
e_sub_u_min = lambda g, X, Y : e_add_u_min(g, X, -Y)
e_div_u_min = lambda g, X, Y : e_mul_u_min(g, X, 1. / Y)

e_add_v_min = lambda g, X, Y : u_add_e_min(g.reverse(), Y, X)
e_mul_v_min = lambda g, X, Y : u_mul_e_min(g.reverse(), Y, X)
e_sub_v_min = lambda g, X, Y : e_add_v_min(g, X, -Y)
e_div_v_min = lambda g, X, Y : e_mul_v_min(g, X, 1. / Y)

v_add_e_min = lambda g, X, Y : e_add_v_min(g, Y, X)
v_mul_e_min = lambda g, X, Y : e_mul_v_min(g, Y, X)
v_sub_e_min = lambda g, X, Y : v_add_e_min(g, X, -Y)
v_div_e_min = lambda g, X, Y : v_mul_e_min(g, X, 1. / Y)

u_add_v = UAddV.apply
u_mul_v = UMulV.apply
u_sub_v = lambda g, X, Y : u_add_v(g, X, -Y)
u_div_v = lambda g, X, Y : u_mul_v(g, X, 1. / Y)
u_dot_v = UDotV.apply

v_add_u = lambda g, X, Y : u_add_v(g.reverse(), Y, X)
v_mul_u = lambda g, X, Y : u_mul_v(g.reverse(), Y, X)
v_sub_u = lambda g, X, Y : v_add_u(g, X, -Y)
v_div_u = lambda g, X, Y : v_mul_u(g, X, 1. / Y)
v_dot_u = lambda g, X, Y : u_dot_v(g.reverse(), Y, X)

# tmp hack
def e_sub_v(g, X, Y):
    return X - copy_u(g.reverse(), Y)

def e_div_v(g, X, Y):
    return X / copy_u(g.reverse(), Y)

def e_mul_v(g, X, Y):
    return X * copy_u(g.reverse(), Y)
