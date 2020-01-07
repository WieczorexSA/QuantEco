# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:22:57 2020

@author: Zuzanka
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import quantecon as qe

alpha = 0.3
beta  = 0.99**40
tau   = 0.0
lambd = 0.5
g     = 0.0 
T  = 50000

zetamean = 1.0
lnzetamean=0.0
rhomean  = 1.0
lnrhomean=0.0
etamean  = 1.0
lnetamean=0.0

std_lnzeta = 0.13
std_lnrho  = 0.50
std_lneta  = 0.95


np.random.seed(seed=123)

#####################1.2 continous ######################

zeta  = np.random.lognormal(mean=lnzetamean, sigma=std_lnzeta, size=T)
rho   = np.random.lognormal(mean=lnrhomean, sigma=std_lnrho, size=T)
eta   = np.random.lognormal(mean=lnetamean, sigma=std_lneta, size=T)

phi = 1/(1+(1-alpha)*(lambd*etamean+tau*(1+lambd*(1-etamean)))/(alpha*zetamean*(1+lambd)))
s=beta*phi/(1+beta*phi)

lnk_ss=math.log((1-alpha)*s*(1-tau)/(1-lambd))/(1-alpha)

lnk = [lnk_ss]

for i in range (1,T):
    lnk.append(math.log((1-alpha)/(1-lambd))+math.log(s)+math.log(1-tau)+math.log(zeta[i])+alpha*lnk[i-1])

lnkmean=np.mean(lnk)

plt.scatter(range(0,T), lnk, label='capital', alpha=0.7, s=0.5)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time- exercise 1.2 continuous case')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()

#------------------------1.2 discrete--------------------------------------------------------

probhighz=0.5
lnzlow=-std_lnzeta*math.sqrt(probhighz/(1-probhighz))
lnzhigh=std_lnzeta*math.sqrt((1-probhighz)/probhighz)

probhighr=0.5
lnrlow=-std_lnrho*math.sqrt(probhighr/(1-probhighr))
lnrhigh=std_lnrho*math.sqrt((1-probhighr)/probhighr)

zeta = np.random.binomial(n=1, p=probhighz, size=T)
zeta=zeta*(-lnzlow+lnzhigh)
zeta=zeta+lnzlow

rho = np.random.binomial(n=1, p=probhighr, size=T)
rho=rho*(-lnrlow+lnrhigh)
rho=rho+lnrlow

eta=qe.quad.qnwnorm(11, 1, std_lneta**2)
np.mean(eta)
np.std(eta)

phih =probhighz*eta[1]/(1+(1-alpha)*(lambd*np.exp(eta[0])+tau*(1+lambd*(1-np.exp(eta[0]))))/(alpha*math.exp(lnzhigh)*(1+lambd)))
phil =(1-probhighz)*eta[1]/(1+(1-alpha)*(lambd*np.exp(eta[0])+tau*(1+lambd*(1-np.exp(eta[0]))))/(alpha*math.exp(lnzlow)*(1+lambd)))
phi=sum(phih)+sum(phil)

s=beta*phi/(1+beta*phi)

lnk_ss=math.log((1-alpha)*s*(1-tau)/(1-lambd))/(1-alpha)

lnk = [lnk_ss]

for i in range (1,T):
    lnk.append(math.log((1-alpha)/(1-lambd))+math.log(s)+math.log(1-tau)+zeta[i]+alpha*lnk[i-1])

lnkmean=np.mean(lnk)

plt.scatter(range(0,T), lnk, label='capital', alpha=0.7, s=0.1)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time - exercise 1.2 discrete case with probability=0.125')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()

#################################1.3-------------------------------------------

















