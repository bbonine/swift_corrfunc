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
path1= "/Users/bbonine/ou/research/corr_func/outputs_2/"

#Specify output directory 
path2 = "/Users/bbonine/ou/research/corr_func/figures/11_3/"


# Read in output file
data = np.genfromtxt(path1+'out.txt', delimiter = ',', unpack=True)
ratios = np.loadtxt(path1+'ratios.txt', delimiter = ',')
x = data[:,0] # bin centers
y = data[:,1] # W(theta)
y_err= np.sqrt(data[:,2]) # error




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
x_fit = np.linspace(66,1200,num = 1000)


'''
%%%%%%%%%%%%%%%%%%
Linear Scale Fit
%%%%%%%%%%%%%%%%%%
'''

plt.figure(figsize = [8,6], dpi = 200)

#Fit
plt.plot(x_fit,powerlaw(x_fit,amp,index), label = 'Fit', color = 'red', linewidth = 1) # fit
plt.errorbar(x,y, yerr = y_err, fmt = 'k^',elinewidth = 0.3, ms = 4, mew = 0.3, mfc = 'none', capsize = 3, label = "SACS", linewidth = 1) # data
plt.text(600,.65, r'$\theta_0 = $' + str(np.around(1/(amp**(1/index)),2)), fontsize = 10) 
plt.text(600,.63, r'$\gamma = $' + str(np.around(1-index,2)),fontsize = 10)

# Koutilidas Result
plt.plot(x_fit,(1/1.6)**(1-1.7)*(x_fit)**(1-1.7), label = 'Koutoulidas et al. ', color = 'red', linewidth = 1, linestyle = '-.')

# Params
plt.title('Best Fit Power Law: All fields')
plt.xlabel('Angular Separation (Arcseconds)', fontsize = 8)
plt.ylabel(r'W$(\theta)$', fontsize = 8)
plt.legend(fontsize = 8)
plt.savefig(path2+ 'fit1_all_bins.png')
plt.close()


'''
%%%%%%%%%%%%%%%%%%
Log Scale Fit
%%%%%%%%%%%%%%%%%%
'''
#Fit
plt.figure(figsize = [8,6], dpi = 200)
plt.loglog(x_fit, powerlaw(x_fit, amp, index), label = 'Fit', color = 'red', linewidth = 1)
plt.errorbar(x, y, yerr=y_err, fmt='k^', elinewidth = 0.3, capsize = 3,ms = 4, mew =0.3, mfc = 'none', label = "SACS")  # Data 

# Koutilidas Result
plt.plot(x_fit, (1/1.6)**(1-1.7)*(x_fit)**(1-1.7), label = 'Koutoulidas et al. ', color = 'red', linewidth = 1, linestyle = '-.')


plt.title('Best Fit Power Law: Log Scale)')  
plt.xlabel('Angular Separation (Arcseconds) [Log-scale]', fontsize = 8)
plt.text(400,.65, r'$\theta_0 = $' + str(np.around(1/(amp**(1/index)),2)), fontsize = 10) 
plt.text(400,.5, r'$\gamma = $' + str(np.around(1-index,2)),fontsize = 10)
plt.ylabel(r'W$(\theta)$', fontsize =8)  
plt.legend(fontsize = 8)    
plt.savefig(path2+ 'fit2_all_bins.png')
plt.close()


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Histogram for Random Sources
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
plt.figure(figsize = [8,6], dpi = 300)
plt.hist((1/ratios), bins =50,  color = '#AAAAFF' , histtype = 'stepfilled', linewidth = 1, alpha = 0.7)
plt.xlabel(r'$\frac{R}{D}$')
plt.ylabel('Counts')
plt.title('Ratio of Simulated to Real Sources in Each Field')
plt.xlim(0,4)
plt.text(2.5, 80, r'$\mu = $' + str(np.around(np.mean(ratios),2)), fontsize = 12)
#plt.savefig(path2+ '/ratios.png')
plt.close()













