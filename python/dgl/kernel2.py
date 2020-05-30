import numpy as np

from ._ffi.function import _init_api
from .base import DGLError
from . import ndarray as nd

def infer_broadcast_shape(shp1, shp2):
    """
    Parameters
    ----------
    shp1 : tuple[int]
    shp2 : tuple[int]

    Returns
    -------
    shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise DGLError("Feature shapes {} and {} are not valid for broadcasting."
                    .format(shp1, shp2))
    return tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))

def u_op_e_sum(op, gidx, X, Y, Z):
    """
    Parameters
    ----------
    op : 'mul' or 'add'
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    Z : out tensor
    """
    _CAPI_DGLKernelUOpESum(op, gidx, X, Y, Z)

def u_op_e_max(op, gidx, X, Y, Z, argX, argY):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul', 'add'

    output
    -------
    Z : (N2, D)
    arg_X : (N2,)
    arg_Y : (N2,)
    """
    _CAPI_DGLKernelUOpEMax(op, gidx, X, Y, Z, argX, argY)

def u_op_e_min(op, gidx, X, Y, Z, argX, argY):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul', 'add'

    output
    -------
    Z : (N2, D)
    arg_X : (N2,)
    arg_Y : (N2,)
    """
    _CAPI_DGLKernelUOpEMin(op, gidx, X, Y, Z, argX, argY)

def copy_u_sum(gidx, X, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Z : out tensor
    """
    _CAPI_DGLKernelCopyUSum(gidx, X, Z)

def copy_u_max(gidx, X, Z, argX):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Z : out tensor
    argX : (N2,) out tensor
    """
    _CAPI_DGLKernelCopyUMax(gidx, X, Z, argX)

def copy_u_min(gidx, X, Z, argX):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Z : out tensor
    argX : (N2,) out tensor
    """
    _CAPI_DGLKernelCopyUMin(gidx, X, Z, argX)

def copy_e_sum(gidx, Y, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)
    Z : out tensor
    """
    _CAPI_DGLKernelCopyESum(gidx, Y, Z)

def copy_e_max(gidx, Y, Z, argY):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)
    Z : out tensor
    """
    _CAPI_DGLKernelCopyEMax(gidx, Y, Z, argY)

def copy_e_min(gidx, Y, Z, argY):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)
    Z : out tensor
    """
    _CAPI_DGLKernelCopyEMin(gidx, Y, Z, argY)

def copy_u(gidx, X, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)
    Z : out tensor
    """
    _CAPI_DGLKernelCopyU(gidx, X, Z)

def u_op_v(op, gidx, X, Y, Z):
    """
    Parameters
    ----------
    op : 'mul', 'add', 'dot'
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (N2, D)

    output
    -------
    Z : (E, D) or (E, 1) if op == 'dot'
    """
    _CAPI_DGLKernelUOpV(op, gidx, X, Y, Z)

_init_api("dgl.kernel2")
