from .. import backend as F

__all__ = ['SPMM_MAP', 'SDDMM_MAP']

SPMM_MAP = {
    ('u_add_e', 'sum') : F.u_add_e_sum,
    ('u_mul_e', 'sum') : F.u_mul_e_sum,
    ('u_sub_e', 'sum') : F.u_sub_e_sum,
    ('u_div_e', 'sum') : F.u_div_e_sum,
    ('u_add_e', 'mean') : F.u_add_e_mean,
    ('u_mul_e', 'mean') : F.u_mul_e_mean,
    ('u_sub_e', 'mean') : F.u_sub_e_mean,
    ('u_div_e', 'mean') : F.u_div_e_mean,
    ('u_add_e', 'max') : F.u_add_e_max,
    ('u_mul_e', 'max') : F.u_mul_e_max,
    ('u_sub_e', 'max') : F.u_sub_e_max,
    ('u_div_e', 'max') : F.u_div_e_max,
    ('u_add_e', 'min') : F.u_add_e_min,
    ('u_mul_e', 'min') : F.u_mul_e_min,
    ('u_sub_e', 'min') : F.u_sub_e_min,
    ('u_div_e', 'min') : F.u_div_e_min,

    ('e_add_u', 'sum') : F.u_add_e_sum,
    ('e_mul_u', 'sum') : F.u_mul_e_sum,
    ('e_sub_u', 'sum') : lambda *args : -F.u_sub_e_sum(*args),
    ('e_div_u', 'sum') : lambda *args : 1. / F.u_div_e_sum(*args),
    ('e_add_u', 'mean') : F.u_add_e_mean,
    ('e_mul_u', 'mean') : F.u_mul_e_mean,
    ('e_sub_u', 'mean') : lambda *args : -F.u_sub_e_mean(*args),
    ('e_div_u', 'mean') : lambda *args : 1. / F.u_div_e_mean(*args),
    ('e_add_u', 'max') : F.u_add_e_max,
    ('e_mul_u', 'max') : F.u_mul_e_max,
    ('e_sub_u', 'max') : lambda *args : -F.u_sub_e_max(*args),
    ('e_div_u', 'max') : lambda *args : 1. / F.u_div_e_max(*args),
    ('e_add_u', 'min') : F.u_add_e_min,
    ('e_mul_u', 'min') : F.u_mul_e_min,
    ('e_sub_u', 'min') : lambda *args : -F.u_sub_e_min(*args),
    ('e_div_u', 'min') : lambda *args : 1. / F.u_div_e_min(*args),

    ('copy_u', 'sum') : F.copy_u_sum,
    ('copy_u', 'mean') : F.copy_u_mean,
    ('copy_u', 'max') : F.copy_u_max,
    ('copy_u', 'min') : F.copy_u_min,
    ('copy_e', 'sum') : F.copy_e_sum,
    ('copy_e', 'mean') : F.copy_e_mean,
    ('copy_e', 'max') : F.copy_e_max,
    ('copy_e', 'min') : F.copy_e_min,
}

SDDMM_MAP = {
    'u_add_v' : F.u_add_v,
    'u_mul_v' : F.u_mul_v,
    'u_dot_v' : F.u_dot_v,
}
