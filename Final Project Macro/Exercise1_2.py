# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#--------------------------Random seed generation------------------------------
np.random.seed(seed=885743368)
#--------------------------Parametrization-------------------------------------
alpha = 0.3
beta  = pow(0.99, 40)
tau   = 0.0
lambd = 0.5
g     = 0.0 
periods  = 50000
exp_zeta = 1.0
exp_rho  = 1.0
exp_eta  = 1.0
std_log_zeta = 0.13
std_log_rho  = 0.50
std_log_eta  = 0.95

#----------Random drawing of the shocks from log-normal distribution-----------
zeta  = np.random.lognormal(mean=math.log(exp_zeta), sigma=std_log_zeta, size=periods)
rho   = np.random.lognormal(mean=math.log(exp_rho), sigma=std_log_rho, size=periods)
eta   = np.random.lognormal(mean=math.log(exp_eta), sigma=std_log_eta, size=periods)

# print(log_eta.mean())

#---------------------------Steady state---------------------------------------
# Phi as in the paper (for tau=0 and mean values of shocks).
phi_ss = (1)/(1+((1-alpha)*(lambd*exp_eta))/(alpha*(1+lambd)*exp_rho))

# Saving rate:
sav_rate_ss = (beta*phi_ss)/(1 + (beta*phi_ss))

# Steady state capital stock (for ln(zeta)=0): 
ln_k_ss = (math.log(sav_rate_ss) + math.log(1-tau) + math.log(1-alpha))/(1-alpha)
k_ss = math.exp(ln_k_ss)
print('Steady state log of capital stock:', ln_k_ss)
print('Steady state capital stock:', k_ss)

#-------------------------------Simulation-------------------------------------
# For tau=0, E(zeta)=E(eta)=E(rho)=1. Phi and saving rate is the same as in the steady state.  

# Definig function for capital path
def capital_path(shock, prev_cap):
    ln_cap = math.log(sav_rate_ss) + math.log(1-tau) + math.log(1-alpha) + math.log(shock) + alpha*prev_cap
    return ln_cap

# Defining initial capital stock 
ln_k=[ln_k_ss]
t=[0]
iter = 0

# Iteration
while iter<periods:
    ln_k.append(capital_path(zeta[iter], ln_k[iter]))
    iter+=1
    t.append(iter)

#--------------------------------Plotting--------------------------------------
    
plt.plot([ln_k_ss]*periods, 'o', label='steady state')
plt.plot(t, ln_k, label='Exercise 1.2', alpha=0.5)
plt.title('Capital path over time (log)')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.show()

#For 

# Defining phi function
#def phi(rho, eta):

 
# Defining saving rate function
