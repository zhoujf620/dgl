def u_op_e_sum_coo(gidx, X, Y, op):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul' or 'add'

    Returns
    -------
    Z : (N2, D)
    """
    pass

def copy_u_sum(gidx, X):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)

    Returns
    -------
    Z : (N2, D)
    """
    pass

def reduce_on_row_coo(gidx, Y):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    Y : (E, D)

    Returns
    -------
    Z : (N2, D)
    """
    pass

def u_op_e_max(gidx, X, Y, op):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul', 'add'

    Returns
    -------
    Z : (N2, D)
    arg_X : (N2, 1)
    arg_Y : (N2, 1)
    """
    pass

def u_op_e_min(gidx, X, Y, op):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (E, D)
    op : 'mul', 'add'

    Returns
    -------
    Z : (N2, D)
    arg_X : (N2, 1)
    arg_Y : (N2, 1)
    """
    pass

def u_op_v(gidx, X, Y, op):
    """
    Parameters
    ----------
    gidx : HeteroGraphIndex (must have only one relation)
    X : (N1, D)
    Y : (N2, D)
    op : 'mul', 'add', 'dot'

    Returns
    -------
    Z : (E, D)
    """
    pass
