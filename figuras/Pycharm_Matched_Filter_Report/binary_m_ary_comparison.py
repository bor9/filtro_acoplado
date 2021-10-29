import numpy as np
from scipy.stats import norm

__author__ = 'ernesto'

# M: number of symbols
M = 2 ** np.arange(1, 6)
# binary error probability
P_be = 4e-6
# r (kbps)
r_b = 9

# gamma_b computation
inv_q = -norm.ppf(P_be * M * np.log2(M) / (2 * (M-1)))
gamma_b = (M ** 2 - 1) / (6 * np.log2(M)) * (inv_q ** 2)
# r computation (kbaudios)
r = r_b / np.log2(M)


print(inv_q)
print(gamma_b)
print(r)


