#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:49:59 2020

Practice with cosmology tools in astroPy. The goal is to use this script
to deproject our angular correlation function produced from the agn.py

@author: bbonine
"""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.special import gamma
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

#Specify working directory
path1= "/Users/bbonine/ou/research/corr_func/outputs_2/"

# Read in outputs:
data = np.genfromtxt(path1+'out.txt', delimiter = ',', unpack=True)
angcorr = data[:,1]
varr = np.sqrt(data[:,2])

gam_fit = 1.5

'''
Reminder: Cosmology values we need
- E(z): cosmo.efunc(z)
-D_a(z):  cosmo.angular_diameter_distance(z)
-dtau(z)/dz: 
    - Tau(z): cosmo.lookback_time(z) # Gyr
    - Integrand: cosmo.lookback_time_integrand(z)
    - From internet: Tau(z) = thubb* integral(0,z)[((1+z)*E(z))^-1 dz]
-thubb: cosmo.hubble_time # Gyr
-c: In units of Mpc/ Gyr? ~ 306.4
'''

# Set cosmology:
cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)





# Define functions from Koutoulidis (2018):
cm = cosmo.Om0 / 0.27
z_med = 2



def j_func(y):
    return (1+y) / cosmo.efunc(y) 

def d_func(y):
    return (1+y) / (cosmo.efunc(y)**3)

def J(z):
    return integrate.quad(j_func,0,z)[0]

# D: "Growig mode of linear pertubations in LCDM"
def D(z):
    return (5*cosmo.Om0*cosmo.efunc(z))/2 * integrate.quad(d_func,z,np.inf)[0]

H = gamma(1/2)*gamma((gam_fit-1)/2) / gamma(gam_fit/2)

def C2(m_h):
    return 1.105*(1+(cm*(m_h / (10**14 / cosmo.h)))**0.255)

def b0(m_h):
    return 0.857*(1+(cm*(m_h / (10**14 / cosmo.h)))**0.55)

def b(z,b0,C2):
    return 1 + ((b0-1)/D(z)) + C2*(J(z) / D(z))


# b~ = b(z) / b0:
def b_tild(z):
    return (1 + ((b0-1)/D(z)) + C2*(J(z) / D(z)))/ b0
J0 = J(0)
D0 = D(0)
# d~ = d(z)/ D0:

