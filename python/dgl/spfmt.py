"""Python interfaces to DGL sparse format controller."""
from ._ffi.function import _init_api

def set_format(format):
    _CAPI_SetKernelFormat(format)

_init_api('dgl.spfmt', __name__)