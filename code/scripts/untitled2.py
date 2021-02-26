#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:17:00 2020

@author: bbonine
"""



'''
Version II: Combining all images with a single correlation computation
'''

# Preform correlation function computation on actual agn data:
# Import Necessary Packages:
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
import os



# Read in AGN table:
# Remote version: cat = "/home/bonine/donnajean/research/agn_corr/data/agntable_total.txt"
cat = '/Users/bbonine/research/ou/corr_func/data/agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
x,y = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(16,17) , unpack = True)


# Read in the flux limit file: 
#lim = '/home/bonine/donnajean/research/agn_corr/data/fluxlimit.txt'
lim = '/Users/bbonine/research/ou/corr_func/data/fluxlimit.txt'
exp, fluxlim = np.loadtxt(lim,skiprows = 1, unpack = True)
exp = np.power(10,exp) #exposure time in log units; convert

# Interpolate the flux values:
func1 = InterpolatedUnivariateSpline(exp,fluxlim) 
xnew = np.linspace(0,10**8, num = 10**7, endpoint = True)
''' Get rid of any duplicates:
#field_list = np.unique(field)
# Select desired AGN in desired field: grb060526, in this case
here = np.where(field == 'grb060124')
x_new = x[here]
y_new = y[here]'''
#index = np.zeros(len(field_list))

#Get rid of any duplicates:
field_list = np.unique(field)

# Create output folder
path1 = "/Users/bbonine/research/ou/corr_func/outputs_stack/"
os.mkdir(path1)



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin Looping through each exposure map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

# Save empty arrays for later. 
num_fields = 20
img_list_x = []
img_list_y = []
rand_list_x = []
rand_list_y = []
rand_list = np.empty((num_fields,1000**2))
for i in range(0,num_fields):
    pixel_angle_sec = (47.1262 / 20)**2 # [square arcseconds]
    pixel_angle_deg = (pixel_angle_sec / 3600**2)
    pix_scale = 47.1262 / 20 # arcseconds / pixel
    # Integrate the broken power law from the paper:
    a = 1.34
    b = 2.37 # +/- 0.01
    f_b = 3.67 * 10 ** (-15) # erg  cm^-2 s^-1
    k = 531.91*10**14 # +/- 250.04; (deg^-2 (erg cm^-2 s^-1)^-1)
    s_ref = 10**-14 # erg cm^-2 s^-1

    def f3(x):
        return ((1/s_ref)**-a)*k*(1/(-a+1))*((f_b**(-a+1))-x**(-a+1)) 

    def f4(x):
        return ((1/s_ref)**-b)*k*((f_b/s_ref)**(b-a))*(-x**(-b+1))
    
    # Read in the relevant exposure map:
    here = np.where(field == field_list[i])
    # Extract source positions in this field:
    img_list_x.append(x[here])
    img_list_y.append(y[here])
    expmap = '/Users/bbonine/research/ou/corr_func/data/grb/'+field[here][0]+'/expo.fits'
    print("Exposure map located")
    # Make directory for outuput files for this field:
    path2 = "/Users/bbonine/research/ou/corr_func/outputs_stack/"+field[here][0]
    os.mkdir(path2)
    print("Directory created...")
    
    # Read in exposure map with astropy
    hdu_list = fits.open(expmap)
    image_data = hdu_list[0].data
    hdu_list.close()
    exp_map_1d =  image_data.ravel() #Conver exposure map to 1D array for later
    
    # Restrict to fields with more than one AGN (necessary for correlation calculation):

    # Save reference pixel value for later
    ref_flux =  image_data[500,500]

    # Use the interpolated function to extract flux limit based off reference flux
    flux_lim = np.asscalar(func1(ref_flux))

    # Find the flux limit for each pixel:
    fluxlimit = np.zeros(len(exp_map_1d))
    for j in range(0,len(fluxlimit)):
        fluxlimit[j] = np.asscalar(func1(exp_map_1d[j]))
        
    fluxlimit_1d = np.asarray(fluxlimit) #convert to numpy array
    fluxlimit_2d = np.reshape(fluxlimit_1d,(-1,len(image_data[0])))

    # Determine number of sources per pixel
    Npix = []
    for j in range(0,len(fluxlimit_1d)):
        if fluxlimit_1d[j] <= f_b:
            Npix.append(f3(fluxlimit_1d[j]) + f4(fluxlimit_1d[j]))
        else:
            Npix.append(f4(fluxlimit_1d[j]))

    N = np.abs(Npix)
    N_source = pixel_angle_deg*N # Number of sources
    N_norm = N_source
    np.max(N_source) # Normalize

    # Construct weight map to gerenate random image:
    weight_map = np.reshape(N_norm,(-1,len(image_data[0])))
    plt.style.use('default')
    plt.figure(figsize = [10, 10])
    plt.imshow(weight_map,cmap = 'gray', interpolation = 'none', origin = 'lower')
    plt.colorbar()
    plt.title('Field '+field[here][0]+ ': Normalized sources per pixel')
    plt.savefig('/Users/bbonine/research/ou/corr_func/outputs_stack/'+field[here][0]+'/expmap.png')
    plt.close()
    print("Exposure map " + str(i+1) + " created..." )

    # Begin making random image:
    weight_tot = np.sum(N_norm) 
    weight_outer = np.cumsum(N_norm) # 'Outer edge' of pixel weight
    weight_inner = weight_outer - N_norm # 'Inner edge' of pixel weight

    n_sources = int(np.sum(N_source)) 
    n_dim = 1000 # specify the dimmension of our image
    img2 = np.zeros(n_dim*n_dim)
    var = np.random.uniform(0,weight_tot,n_sources)
    for l in range(0,n_sources):
        for m in range(0,len(img2)):
            if var[l] > weight_inner[m] and var[l] < weight_outer[m]:
                img2[m] = img2[m] + 1 # specifies flux of pixel. 

    # Save random image to file:
    rand_img = np.reshape(img2,(n_dim,n_dim)) 
    here2 = np.where(rand_img > 0)
    rand_list_x.append(here2[0]) # image position of x values
    rand_list_y.append(here2[1]) # image position of y vales
    
# Gather master list of data point positions
data_x = []
data_y = []
rand_x = []
rand_y = []

for i in range(0,len(img_list_x)):
    for j in range(0,len(img_list_x[i])):
       data_x.append(img_list_x[i][j])
       data_y.append(img_list_y[i][j])

for i in range(0,len(rand_list_x)):
    for j in range(0,len(rand_list_x[i])):
        rand_x.append(rand_list_x[i][j])
        rand_y.append(rand_list_y[i][j])
    
    
# Define pixel distance function between two sources:
def distance(x2,x1,y2,y1):
    return (((x2-x1)**2 + (y2-y1)**2)**0.5)
dist_rr = []
for j in range(len(rand_x)):
    for k in range(len(rand_x)):
        if k != j:
            dist_rr.append(distance(rand_x[k],rand_x[j],rand_y[k],rand_y[j]))
            
rr_ang_dist = pix_scale * np.asarray(dist_rr) # arcsec/pix * pix = arcsec

# Repeat same process for data-data:
dist_dd = []
for j in range(len(data_x)):
    for k in range(len(data_x)):
        if k != j:
            dist_dd.append(distance(data_x[k],data_x[j],data_y[k],data_y[j]))
dd_ang_dist = pix_scale * np.asarray(dist_dd)

# And data-random:
dist_dr = []
for j in range(len(data_x)):
    for k in range(len(rand_x)):
        if k != j:
            dist_dr.append(distance(rand_x[k],data_x[j],rand_y[k],data_y[j]))
dr_ang_dist = pix_scale * np.asarray(dist_dr)


# Bin the data:
dd_binned = np.histogram(dd_ang_dist)
bins = dd_binned[1] # selects the 'bins' array from np.histogram

# Select the angular separations

dd = np.histogram(dd_ang_dist, bins = bins)[0]
dr = np.histogram(dr_ang_dist, bins = bins)[0]
rr = np.histogram(rr_ang_dist, bins = bins)[0]

 
# Begin calculating angular correlation function:
N_d = len(data_x)
N_r = len(rand_x)
N = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))

# Hamilton correlation estimator
def W(DD,DR,RR):
    return (N *((DD * RR) / (DR)**2) ) -1

        
# Compute Correlation function for whole data set
corr = []
for j in range(0,len(dr)):
     corr.append(W(dd[j],dr[j],rr[j]))

#Varience:
varr = []
for j in range(0,len(dr)):
    varr.append(3*(1+(W(dd[j],dr[j],rr[j])))**2 / dd[j])
    
# Plot results
plt.style.use('default')
plt.figure(figsize = [12, 8])
plt.errorbar(bins[0:10],corr, yerr = np.sqrt(varr), fmt = '.')
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('All fields: Correlation Function')
plt.savefig('/Users/bbonine/research/ou/corr_func/outputs_stack/corr_func.png')
plt.close()
   
    
# Save stacked images to file  
#random 
plt.style.use('dark_background')
plt.figure(figsize = [12, 12])
plt.scatter(rand_x,rand_y, marker = '.', color = 'white', s = 25)
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.title('Random Image: Stacked')
plt.savefig('/Users/bbonine/research/ou/corr_func/outputs_stack/rand_img.png')
plt.close()

# Save data image to file:
plt.style.use('dark_background')
plt.figure(figsize = [12, 12])
plt.scatter(data_x,data_y, marker = '.', color = 'white', s = 25)
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.title('Data: Stacked')
plt.savefig('/Users/bbonine/research/ou/corr_func/outputs_stack/data_img.png')
plt.close()






    
print("Correlation analysis complete! Have a nice day.")    