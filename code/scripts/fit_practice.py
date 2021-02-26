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
path = '/Users/bbonine/ou/research/corr_func/figures/02_20_21_data_log/'


# Read in output files
centers,corr,sig = np.loadtxt(path+'out.txt',usecols = (0,1,2), skiprows = 1, unpack = True)


#ratios = np.genfromtxt(path1+'ratio.txt',delimiter = ',')

# Select only positive values to use in the fit
pos_vals = np.where(corr > 0)[0]

x = centers[pos_vals]
y = corr[pos_vals]
y_err = sig[pos_vals]
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
amp = 10.0**(pfinal[0])

indexErr = np.sqrt(covar[1][1])
ampErr = np.sqrt(covar[0][0]) * amp

#Plot the data

# Set global plot params


fontsize = 10
figsize = (8,6)
dpi = 300

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
x_fit = np.linspace(1,700,num = 1000)


'''
%%%%%%%%%%%%%%%%%%
Linear Scale Fit
%%%%%%%%%%%%%%%%%%
'''

plt.figure(figsize = [8,6], dpi = 300)

#Fit
plt.plot(x_fit,powerlaw(x_fit,amp,index), label = 'SACS', color = 'red', linewidth = 0.6) # fit
plt.errorbar(centers,corr, yerr = sig, fmt = 'ko',elinewidth = 0.3, ms = 4, mew = 0.3, mfc = 'none', capsize = 3, linewidth = 0.6) # data
# Koutilidas Result
plt.plot(x_fit,(1/1.6)**(1-1.7)*(x_fit)**(1-1.7), label = 'Koutoulidas et al. ', color = 'red', linewidth = 0.6, linestyle = '-.')
# Params
plt.title('Best Fit Power Law: All fields')
plt.xlabel('Angular Separation (Pixels)', fontsize = 8)
plt.ylabel(r'W$(\theta)$', fontsize = 8)
plt.legend(fontsize = 8)

plt.xscale('log')
plt.yscale('log')
plt.text(300,0.6, r'$\theta_0 = $' + str(np.around(1/(amp**(1/index)),3)), fontsize = 10)
plt.text(300,1.1, r'$\gamma = $' + str(np.around(1-index,3)),fontsize = 10)

plt.xlim(10,650)
plt.savefig(path+ 'fit_lin1.png')
plt.close()


'''
%%%%%%%%%%%%%%%%%%
Log Scale Fit
%%%%%%%%%%%%%%%%%%

#Fit
plt.figure(figsize = [8,6], dpi = 200)
plt.loglog(x_fit, powerlaw(x_fit, amp, index), label = 'Fit', color = 'red', linewidth = 1)
plt.errorbar(centers, corr, yerr=sig, fmt='k^', elinewidth = 0.3, capsize = 3,ms = 4, mew =0.3, mfc = 'none', label = "SACS")  # Data

# Koutilidas Result
plt.plot(x_fit, (1/1.6)**(1-1.7)*(x_fit)**(1-1.7), label = 'Koutoulidas et al. ', color = 'red', linewidth = 1, linestyle = '-.')


plt.title('Best Fit Power Law: Log Scale)')
plt.xlabel('Angular Separation (Pixels) [Log-scale]', fontsize = 8)
plt.text(400,.65, r'$\theta_0 = $' + str(np.around(1/(amp**(1/index)),2)), fontsize = 10)
plt.text(400,.5, r'$\gamma = $' + str(np.around(1-index,2)),fontsize = 10)
plt.ylabel(r'W$(\theta)$', fontsize =8)
plt.legend(fontsize = 8)
plt.savefig(path+ 'fit_lin2.png')
plt.close()
'''

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Histogram for Random Sources
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure(figsize = [7,6], dpi = 300)
plt.hist(ratios, bins =60,  color = '#AAAAFF' , histtype = 'stepfilled', linewidth = 1, alpha = 0.7)

plt.vlines(0.5,0,160, linestyle = '--', linewidth = 0.5 )
plt.vlines(2.5,0,160, linestyle = '--', linewidth = 0.5 )

plt.xlabel(r'$\frac{D}{R}$')
plt.ylabel('Counts')
plt.title('SACS: All Fields')
plt.xlim(0,4)
plt.ylim(0,160)
plt.text(3, 120, r'$\mu = $' + str(np.around(np.mean(ratios),2)), fontsize = 12)
plt.savefig(path2+ 'ratios.png')
plt.close()

'''


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Histogram for Random Sources; cut
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


here = np.where(1/ratios < 2)
plt.figure(figsize = [8,6], dpi = 300)
plt.hist((1/ratios[here]), bins =20,  color = '#AAAAFF' , histtype = 'stepfilled', linewidth = 1, alpha = 0.7)
plt.xlabel(r'$\frac{R}{D}$')
plt.ylabel('Counts')
plt.title('Ratio of Simulated to Real Sources in Each Field: Reduced Sample')
plt.xlim(0,2)
plt.text(1.5, 80, r'$N_{fields} = $' + str(len(here[0])), fontsize = 8)
plt.text(1.5, 75, r'$\mu = $' + str(np.around(np.mean(1/ratios[here]),2)), fontsize = 8)
plt.savefig(path2+ '/ratios_cut.png')
plt.close()

'''
