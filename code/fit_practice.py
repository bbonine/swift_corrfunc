#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:59:00 2020
Going to use this script to practice fitting some functions. 
This script follow sthe scipy cookbook avaliable here: https://scipy-cookbook.readthedocs.io/items/FittingData.html
@author: bbonine
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#Specify working directory
path1= "/Users/bbonine/ou/research/ou/corr_func/outputs_2/"

# Read in output file
data = np.genfromtxt(path1+'out.txt', delimiter = ',', unpack=True)
ratios = np.loadtxt(path1+'ratios.txt', delimiter = ',')
x = data[0:-1,0] # bin centers
y = data[0:-1,1] # W(theta)
y_err= np.sqrt(data[0:-1,2]) # error




#convert to log

logx = np.log10(x)
logy = np.log10(y)
logyerr = (1/2.3)* y_err / y

#Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

# define our line fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p,x)) / err

pinit = [1.0, -1.0]
out = optimize.leastsq(errfunc, pinit, 
                       args = (logx, logy, logyerr), full_output = 1)
pfinal = out[0]
covar = out[1]
print(pfinal)
print(covar)

index = pfinal[1]
amp = 10.0**pfinal[0]

indexErr = np.sqrt(covar[1][1])
ampErr = np.sqrt(covar[0][0]) * amp

#Plot the data

# Set global plot params 


fontsize = 20
figsize = (8,6)
dpi = 150

plt.rcParams.update({'font.size': fontsize, 'figure.figsize': figsize, 'figure.dpi': dpi})
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2



plt.style.use('default')
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize = [8,8])
plt.figure(dpi = 300)
x_fit = np.linspace(66,1000,num = 1000)

plt.subplot(2,1,1)
plt.plot(x_fit,powerlaw(x_fit,amp,index), label = 'Fit', color = 'red') # fit
plt.errorbar(x,y, yerr = y_err, fmt= 'k.', capsize = 5, label = "Fields 1 - 739") # data
plt.text(500,.2, 'Ampli = %5.2f +/- %5.2f' % (amp,ampErr), fontsize = 6) 
plt.text(500,.18, 'Index = %5.2f +/- %5.2f' % (index, indexErr),fontsize = 6)
plt.title('Best Fit Power Law: All fields')
plt.xlabel('Angular Separation (Arcseconds)', fontsize = 12)
plt.ylabel(r'W$(\theta)$', fontsize = 12)
plt.legend(fontsize = 8)

plt.subplot(2, 1, 2)
plt.loglog(x_fit, powerlaw(x_fit, amp, index), label = 'Fit', color = 'red')
plt.errorbar(x, y, yerr=y_err, fmt='k.', capsize = 5, label = "Fields 1 - 739")  # Data   
plt.xlabel('Angular Separation (Arcseconds) [Log-scale]', fontsize = 12)
plt.ylabel(r'W$(\theta)$', fontsize = 12)  
plt.legend(fontsize = 8)    
plt.tight_layout()
plt.savefig(path1+ 'fit.png')
plt.close()



plt.figure(dpi = 300)
plt.hist((1/ratios), bins =50,  color = 'limegreen' , histtype = 'step', linewidth = 1)
plt.xlabel(r'$\frac{R}{D}$')
plt.ylabel('Counts')
plt.title('Ratio of Simulated to Real Sources in Each Field')
plt.xlim(0,4)
plt.text(3.5, 110, r'$\mu = 1.3 $', fontsize = 10)
plt.savefig(path1+ '/ratios.png')
plt.close()













