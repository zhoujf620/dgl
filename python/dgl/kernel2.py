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
    pass

def u_op_e_sum(op, gidx, X, Y, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul' or 'add'
    Z : out tensor
    """
    pass

def copy_u_sum(gidx, X, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Z : out tensor
    """
    pass

def copy_e_sum(gidx, Y, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)
    Z : out tensor
    """
    pass

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
    pass

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
    pass

def u_op_v(op, gidx, X, Y, Z):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (N2, D)
    op : 'mul', 'add', 'dot'

    output
    -------
    Z : (E, D) or (E, 1) if op == 'dot'
    """
    pass
