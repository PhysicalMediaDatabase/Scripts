import numpy as np

# Convert an array of non-linear signal values (N) to an array of 
# linear light values (L) using the ST 2084 EOTF (PQ curve)
def eotf_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    L = (np.clip(np.power(N, (1/m2)) - c1, 0, None) / (c2 - c3 * np.power(N, (1/m2))))
    return np.nan_to_num(np.power(L, 1/m1), nan=0.0)

# Convert an array of linear light values (L) to an array of
# non-linear signal values (N) using the inverse ST 2084 EOTF (PQ curve)
def eotf_inverse_ST2084(N, m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875):
    Y_p = np.power(N, m1)
    return np.power((c1 + c2 * Y_p) / (c3 * Y_p + 1), m2)