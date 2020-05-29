import torch as th
from ... import kernel2 as K

class UMulESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_e_sum('mul', g, X, Y, Z)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = u_mul_e_sum(K.transpose(g), dZ, Y)
            dX = reduce_on_broadcast_dim(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = u_mul_v(g, X, Y)
            dY = reduce_on_broadcast_dim(dY, Y.shape)
        return None, dX, dY

class UAddESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_e_sum('add', g, X, Y, Z)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = copy_u_sum(K.transpose(g), dZ)
            dX = reduce_on_broadcast_dim(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = row_to_nonzero(K.transpose(g), dZ)
            dY = reduce_on_broadcast_dim(dY, Y.shape)
        return None, dX, dY

class UMulEMax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_max('mul', g, X, Y, Z, argX, argY)
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
            deltaX = reduce_on_broadcast_dim(Y[argY] * dZ, X.shape)
            dX = th.scatter_add(dX, 0, argX, deltaX)
        if ctx.needs_input_grad[2]:
            dY = th.zeros_like(Y)
            dY[argY] = reduce_on_broadcast_dim(X[argX] * dZ, Y.shape)
        return None, dX, dY

class UMulEMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_min('mul', g, X, Y, Z, argX, argY)
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
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_max('add', g, X, Y, Z, argX, argY)
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
            deltaX = reduce_on_broadcast_dim(dZ, X.shape)
            dX = th.scatter_add(dX, 0, argX, deltaX)
        if ctx.needs_input_grad[2]:
            dY = th.zeros_like(Y)
            dY[argY] = reduce_on_broadcast_dim(dZ, Y.shape)
        return None, dX, dY

class UAddEMin(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_nodes(1),) + bshape, device=X.device, dtype=X.dtype)
        argX = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        argY = th.zeros((Z.size(0),), device=X.device, dtype=getattr(th, g.dtype))
        K.u_op_e_min('add', g, X, Y, Z, argX, argY)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y, argX, argY)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return UAddEMax.backward(ctx, dZ)

class RowToNonZero(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        Z = th.zeros((g.number_of_edges(0), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        K.row_to_nonzero(g, X, Z)
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dX = None
        if ctx.needs_input_grad[1]:
            dX = copy_e_sum(adj, dZ)
        return None, dX

class CopyESum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, Y):
        Z = th.zeros((g.number_of_nodes(1), ) + Y.shape[1:], device=Y.device, dtype=Y.dtype)
        K.copy_e_sum(g, Y, Z)
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dY = None
        if ctx.needs_input_grad[1]:
            dY = row_to_nonzero(g, dZ)
        return None, dY


class CopyUSum(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X):
        Z = th.zeros((g.number_of_nodes(1), ) + X.shape[1:], device=X.device, dtype=X.dtype)
        K.copy_u_sum(g, X, Z)
        ctx.backward_cache = g
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dX = None
        if ctx.needs_input_grad[1]:
            dX = copy_u_sum(K.transpose(g), dZ)
        return None, dX

class UMulV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_edges(0),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_v('mul', g, X, Y, Z)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = u_mul_e_sum(K.transpose(g), Y, dZ)
            dX = reduce_on_broadcast_dim(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = u_mul_e_sum(g, X, dZ)
            dY = reduce_on_broadcast_dim(dY, Y.shape)
        return None, dX, dY

class UAddV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:], Y.shape[1:])
        Z = th.zeros((g.number_of_edges(0),) + bshape, device=X.device, dtype=X.dtype)
        K.u_op_v('add', g, X, Y, Z)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        g = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[1]:
            dX = copy_e_sum(K.transpose(g), dZ)
            dX = reduce_on_broadcast_dim(dX, X.shape)
        if ctx.needs_input_grad[2]:
            dY = copy_e_sum(g, dZ)
            dY = reduce_on_broadcast_dim(dY, Y.shape)
        return None, dX, dY

class UDotV(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, X, Y):
        bshape = K.infer_broadcast_shape(X.shape[1:-1], Y.shape[1:-1])
        Z = th.zeros((g.number_of_edges(0),) + bshape + (1,), device=X.device, dtype=X.dtype)
        K.u_op_v('dot', g, X, Y, Z)
        ctx.backward_cache = g
        ctx.save_for_backward(X, Y)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        return UMulV.backward(ctx, dZ)

copy_e_sum = CopyESum.apply
copy_u_sum = CopyUSum.apply
row_to_nonzero = RowToNonZero.apply

u_add_e_sum = UAddESum.apply
u_mul_e_sum = UMulESum.apply
u_sub_e_sum = lambda g, X, Y : u_add_e_sum(g, X, -Y)
u_div_e_sum = lambda g, X, Y : u_mul_e_sum(g, X, 1. / Y)

e_add_u_sum = lambda g, X, Y : u_add_e_sum(g, Y, X)
e_mul_u_sum = lambda g, X, Y : u_mul_e_sum(g, Y, X)
e_sub_u_sum = lambda g, X, Y : e_add_u_sum(g, X, -Y)
e_div_u_sum = lambda g, X, Y : e_mul_u_sum(g, X, 1. / Y)

e_add_v_sum = lambda g, X, Y : u_add_e_sum(K.transpose(g), Y, X)
e_mul_v_sum = lambda g, X, Y : u_mul_e_sum(K.transpose(g), Y, X)
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

e_add_v_max = lambda g, X, Y : u_add_e_max(K.transpose(g), Y, X)
e_mul_v_max = lambda g, X, Y : u_mul_e_max(K.transpose(g), Y, X)
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

e_add_v_min = lambda g, X, Y : u_add_e_min(K.transpose(g), Y, X)
e_mul_v_min = lambda g, X, Y : u_mul_e_min(K.transpose(g), Y, X)
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

v_add_u = lambda g, X, Y : u_add_v(K.transpose(g), Y, X)
v_mul_u = lambda g, X, Y : u_mul_v(K.transpose(g), Y, X)
v_sub_u = lambda g, X, Y : v_add_u(g, X, -Y)
v_div_u = lambda g, X, Y : v_mul_u(g, X, 1. / Y)
v_dot_u = lambda g, X, Y : u_dot_v(K.transpose(g), Y, X)

# tmp hack
def e_sub_v(g, X, Y):
    return X - row_to_nonzero(K.transpose(g), Y)

def e_div_v(g, X, Y):
    return X / row_to_nonzero(K.transpose(g), Y)

def e_mul_v(g, X, Y):
    return X * row_to_nonzero(K.transpose(g), Y)
