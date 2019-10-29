# Question 2 - redoing Question 1. for different gamma

import sympy as sp
import numpy as np
import matplotlib as mpl
import scipy as scp
import time
import random

random.seed(42693)


##### Part 1 - no correlation

mean = [1, 1]
cov = [[1, 0], [0, 1]] #we need to have identity matrix as a variance-covariance matrix to assure no correlation
log_k, log_z = np.random.multivariate_normal(mean, cov, 10**7).T

# Plotting k,z in logs

mpl.pyplot.scatter(log_k, log_z, s=1, color='blue')
mpl.pyplot.show()

k = np.exp(log_k)
z = np.exp(log_z)

# Plotting k,z in levels

mpl.pyplot.scatter(k, z, s=1, color='blue')
mpl.pyplot.show()

# 2)

gamma = 0.8

def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod = y(k, z)

#3)
# Idk what's going on in this exercise
agg_prod = sum(prod)
capital = sum(k)

#4)
# I guess it's like picking k's to match a fixed order of z's in such order that the sum of y's will be maximal
#sort k, z from highest to lowest

k_opt = np.array(sorted(k, reverse = True))
z_opt = np.array(sorted(z, reverse = True))

prod_opt = y(k_opt, z_opt)
agg_prod_opt = sum(prod_opt)

#5)
# Reallocation problem

prod_gain = (agg_prod_opt/agg_prod - 1) * 100

# For the purpose of Ex. 3.1
np.var(log_k)
np.var(log_z)

##### Part 2 - correlation +0.5

# 1)

mean = [1, 1]
cov = [[1, 0.5], [0.5, 1]] #we need to have identity matrix as a variance-covariance matrix to assure no correlation
log_k, log_z = np.random.multivariate_normal(mean, cov, 10**7).T

# Plotting k,z in logs

mpl.pyplot.scatter(log_k, log_z, s=1, color='blue')
mpl.pyplot.show()

k = np.exp(log_k)
z = np.exp(log_z)

# Plotting k,z in levels

mpl.pyplot.scatter(k, z, s=1, color='blue')
mpl.pyplot.show()

# 2)

gamma = 0.8

def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod = y(k, z)

#3)
# Idk what's going on in this exercise
agg_prod = sum(prod)
capital = sum(k)

#4)
# I guess it's like picking k's to match a fixed order of z's in such order that the sum of y's will be maximal
#sort k, z from highest to lowest

k_opt = np.array(sorted(k, reverse = True))
z_opt = np.array(sorted(z, reverse = True))

prod_opt = y(k_opt, z_opt)
agg_prod_opt = sum(prod_opt)

#5)
# Reallocation problem

prod_gain = (agg_prod_opt/agg_prod - 1) * 100

# For the purpose of Ex. 3.1
np.var(log_k)
np.var(log_z)

##### Part 3 - correlation -0.5

# 1)

mean = [1, 1]
cov = [[1, -0.5], [-0.5, 1]] #we need to have identity matrix as a variance-covariance matrix to assure no correlation
log_k, log_z = np.random.multivariate_normal(mean, cov, 10**7).T

# Plotting k,z in logs

mpl.pyplot.scatter(log_k, log_z, s=1, color='blue')
mpl.pyplot.show()

k = np.exp(log_k)
z = np.exp(log_z)

# Plotting k,z in levels

mpl.pyplot.scatter(k, z, s=1, color='blue')
mpl.pyplot.show()

# 2)

gamma = 0.8

def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod = y(k, z)

#3)
# Idk what's going on in this exercise
agg_prod = sum(prod)
capital = sum(k)

#4)
# I guess it's like picking k's to match a fixed order of z's in such order that the sum of y's will be maximal
#sort k, z from highest to lowest

k_opt = np.array(sorted(k, reverse = True))
z_opt = np.array(sorted(z, reverse = True))

prod_opt = y(k_opt, z_opt)
agg_prod_opt = sum(prod_opt)

#5)
# Reallocation problem

prod_gain = (agg_prod_opt/agg_prod - 1) * 100


# For the purpose of Ex. 3.1
np.var(log_k)
np.var(log_z)