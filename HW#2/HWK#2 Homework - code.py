# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:15:31 2019

@author: szwie
"""

## Exercise 1 - Taylor approximation

import math
import sympy as sy
import numpy as np
import numpy.polynomial.chebyshev as cheby
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Factorial function
def factorial(n):
    k = 1
    for i in range(n):
        k = k * (i + 1)
    return k

# Taylor approximation at x0 of the function 'function'


def taylor(function, x0, n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x, i).subs(x, x0))/(factorial(i))*(x - x0)**i
        i += 1
    return p
    
x = sy.Symbol('x')
f= x**(0.321)

def plot():
    x_lims = [0,4]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    # Use it up to 1, 2, 5 and 20 order approximations. 
    order = [1,2,5,20]
    for j in order:
        func = taylor(f,1,j)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='Order '+str(j))
        y1 = []
   
    # Plot the function to approximate
    plt.plot(x1,x1**(0.321),label=r'$x^{0.321}')
    plt.xlim(x_lims)
    plt.ylim([-2,7])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation for x^0.321')
    plt.show()

plot()

## Exercise 2 - Approximation of the ramp function with Taylor series

x = sy.Symbol('x', real = True)
ramp = (x+abs(x))*0.5

def plot():
    x_lims = [-2,6]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    # Use it up to 1, 2, 5 and 20 order approximations. 
    order = [1,2,5,20]
    for j in order:
        func = taylor(ramp,2,j)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='Order '+str(j))
        y1 = []
   
    # Plot the function to approximate
    plt.plot(x1,(x1+abs(x1))*0.5,label=r'(x+|x|)/2')
    plt.xlim(x_lims)
    plt.ylim([-2,6])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation - ramp function')
    plt.show()

plot()

## plotting the 

## Exercise 3 - Approximation of e^(1/x), runge function and ramp function


x = sy.Symbol('x')
expon = math.e

def f(x):
    with np.errstate(divide='ignore', invalid='ignore'):
       return expon**(1/x)

x_lims=[-1,1]
x1 = np.linspace(x_lims[0],x_lims[1],11) 
yf= f(x1)
yf[yf == inf] = 2000

xf = np.linspace(x_lims[0],x_lims[1],101)
cf = poly.polyfit(x1,yf,3)
ffit3 = poly.polyval(xf,cf)
# Monomials of order 5:
coeffs = poly.polyfit(x1,yf,5)
ffit = poly.polyval(xf,coeffs)

# Monomials of order 10: 
coeffs10 = poly.polyfit(x1,yf,10)
ffit10 = poly.polyval(xf,coeffs10)

def g(x):
	return 1.0 / (1.0 + 25.0 * x**2)
yg = g(x1)

cg = poly.polyfit(x1,yg,3)
gfit3 = poly.polyval(xf,cg)

# Monomials of order 5:
coeffg = poly.polyfit(x1,yg,5)
gfit = poly.polyval(xf,coeffg)

# Monomials of order 10: 
coeffg10 = poly.polyfit(x1,yg,10)
gfit10 = poly.polyval(xf,coeffg10)

def h(x):
	return  (x+abs(x))*0.5
yh = h(x1)

ch = poly.polyfit(x1,yh,3)
hfit3 = poly.polyval(xf,ch)

# Monomials of order 5:
coeffh = poly.polyfit(x1,yh,5)
hfit= poly.polyval(xf,coeffh)

# Monomials of order 10: 
coeffh10 = poly.polyfit(x1,yh,10)
hfit10 = poly.polyval(xf,coeffh10)

fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot(321)
plt.plot(x1,yf,'o',label='data')
plt.plot(xf,f(xf), label='True')
plt.plot(xf,ffit3,label='Cubic Polynomials') 
plt.plot(xf,ffit,label='Monomial of order 5')
plt.plot(xf,ffit10,label='Monomial of order 10')
plt.legend(bbox_to_anchor=(0.5,0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
plt.ylim(-0.2,1)
plt.title('Runge Function $1 / (1 + 25x^2) $')

plt.subplot(322, sharex=ax1)
plt.plot(xf,f(xf)-ffit3,label='Cubic error') 
plt.plot(xf,f(xf)-ffit,label='m5 error')
plt.plot(xf,f(xf)-ffit10,label='m10 error')
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
plt.title('Runge Function Approximation Errors')

plt.subplot(323, sharex=ax1)
plt.plot(x1,yg,'o')
plt.plot(xf,g(xf))
plt.plot(xf,gfit3) 
plt.plot(xf,gfit)
plt.plot(xf,gfit10)
plt.ylim(-100,1000)
plt.title('Exponential Function $exp(1/x)$')

plt.subplot(324, sharex=ax1)
plt.plot(xf,g(xf)-gfit3) 
plt.plot(xf,g(xf)-gfit)
plt.plot(xf,g(xf)-gfit10)
plt.ylim(0,100)
plt.title('Exponential Function Approximation Errors')


plt.subplot(325, sharex=ax1)
plt.plot(x1,yh,'o')
plt.plot(xf,h(xf))
plt.plot(xf,hfit3) 
plt.plot(xf,hfit)
plt.plot(xf,hfit10)
plt.ylim(-0.5,2.5)
plt.title('Ramp Function $(x+|x|)/2$')

plt.subplot(326, sharex=ax1)
plt.plot(xf,h(xf)-hfit3) 
plt.plot(xf,h(xf)-hfit)
plt.plot(xf,h(xf)-hfit10)
plt.ylim(0,0.25)
plt.title('Ramp Function Approximation Errors')

plt.xlim(x_lims)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.show()
'''
def plot():
    x_lims = [-1,1]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    # Use it up to 1, 2, 5 and 20 order approximations. 
    order = [3,5,10]
    for j in order:
        func = taylor(exponential,1,j)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='Order '+str(j))
        y1 = []
   
    # Plot the function to approximate
    plt.plot(x1,expon**(1/x1),label=r'e^(1/x)')
    plt.xlim(x_lims)
    plt.ylim([-10,50])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation - e^(1/x) function')
    plt.show()

plot()

a = np.array([-1,-0.5,0.5,1])
b = np.array([expon**(-1),expon**(-2),expon**(2),expon])
c = poly.polyfit(a,b,3)
c

poly.polyval([-8.10469283, -2.95882008,  9.27989402],1)

poly.polynomial.chebyshev.chebval(a,b,tensor=True)

#Chebyshev



runge = 1/(1+25*pow(x,2))

def plot():
    x_lims = [-1,1]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    # Use it up to 1, 2, 5 and 20 order approximations. 
    order = [3,5,10]
    for j in order:
        func = taylor(runge,0,j)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='Order '+str(j))
        y1 = []
   
    # Plot the function to approximate
    plt.plot(x1,1/(1+25*x1**(2)),label=r'1/(1+25*x^2)')
    plt.xlim(x_lims)
    plt.ylim([-2,2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation - runge function')
    plt.show()

plot()

a = np.array([-1,-0.5,0,0.5,1])
b = np.array([1/26,1/7.25,1,1/7.25,1/26])
c = np.polyfit(a, b, 3)
c

np.polyval([-2.16884054e-16, -6.06290261e-01,  1.40433339e-16],1)

np.polynomial.chebyshev.chebval(a,b,tensor=True)
#Chebyshev 

ramp = (x+abs(x))*0.5

def plot():
    x_lims = [-1,1]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    # Use it up to 1, 2, 5 and 20 order approximations. 
    order = [3,5,10]
    for j in order:
        func = taylor(ramp,1,j)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='Order '+str(j))
        y1 = []
   
    # Plot the function to approximate
    plt.plot(x1,(x1+abs(x1))*0.5,label=r'(x+|x|)/2')
    plt.xlim(x_lims)
    plt.ylim([-2,6])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation - ramp function')
    plt.show()
    
plot()

a = np.array([-1,-0.5,0,0.5,1])
b = np.array([0,0,0,0.5,1])
c = np.polyfit(a, b, 3)
c

np.polyval([-6.44882017e-16,  4.28571429e-01,  5.00000000e-01],1)

np.polynomial.chebyshev.chebval(a,b,tensor=True)
#Chebyshev

### later polynomials and stuff like that

## Question 2 - Multivariate Function Approximation

alpha = 0.5
sigma = 0.25
k = sy.Symbol('k')
h = sy.Symbol('h')
k in range(0,10)
h in range(0,10)
ces =( ((1-alpha)*k**((sigma -1)/sigma))+alpha*h**((sigma-1)/sigma))**(sigma/(sigma-1))

# display CES
es = ces.diff(ces,k)*(k/ces)


'''

'''

np.polynomial.chebyshev.chebval(np.array[0,10,0.01],np.array[0,10,0.01])

'''