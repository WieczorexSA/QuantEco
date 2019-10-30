# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:37:18 2019

@author: szwie
"""

# Question 3

# 1

import sympy as sp
import numpy as np
import matplotlib as mpl
import scipy as scp
import time
import random

log_k = np.random.normal(-0.5,1,10**4)
log_z = np.random.normal(-0.5,1,10**4)

np.var(log_k)
np.var(log_z)

# 2

gamma = 0.6

k = np.exp(log_k)
z = np.exp(log_z)

def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod = y(k, z)

agg_prod = sum(prod)
capital = sum(k)

k_opt = np.array(sorted(k, reverse = True))
z_opt = np.array(sorted(z, reverse = True))

prod_opt = y(k_opt, z_opt)
agg_prod_opt = sum(prod_opt)

prod_gain = (agg_prod_opt/agg_prod - 1) * 100

# 3 - 1000 iterations of # 2

N = 1000
prod_change = []

for i in range(N):
    log_k = np.random.normal(-0.5,1,10**4)
    log_z = np.random.normal(-0.5,1,10**4)
    k = np.exp(log_k)
    z = np.exp(log_z)
    prod = y(k, z)
    agg_prod = sum(prod)
    capital = sum(k)
    k_opt = np.array(sorted(k, reverse = True))
    z_opt = np.array(sorted(z, reverse = True))
    prod_opt = y(k_opt, z_opt)
    agg_prod_opt = sum(prod_opt)
    prod_gain = (agg_prod_opt/agg_prod - 1) * 100
    prod_change.append(prod_gain)

mpl.pyplot.hist(prod_change, 10, facecolor = 'green', alpha = 0.5)
mpl.pyplot.show()

prod_10k_mean = np.mean(prod_change)
prod_10k_median = np.median(prod_change)
prod_10k_std = np.std(prod_change)

# 4 - Probability of 

log_k = np.random.normal(-0.5,1,10**7)
log_z = np.random.normal(-0.5,1,10**7)

#iterate all the code above

prod_10M_mean = np.mean(prod_change)
prod_10M_median = np.median(prod_change)
prod_10M_std = np.std(prod_change)

#iterate the code for 1-3

prod_cg_range = np.linspace(prod_10M_mean - 0.1*prod_10M_std, prod_10M_mean + 0.1*prod_10M_std,2)
prod_cg_ind = []

for i in range(N):
    if prod_change[i] >= prod_cg_range[0]:
        if prod_change <= prod_cg_range[1]:
            prod_cg_ind.append(1)
        else:
            prod_cg_ind.append(0)
    else:
        prod_cg_ind.append(0)

        


