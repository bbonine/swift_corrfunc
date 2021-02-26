#!/usr/bin/env python
# coding: utf-8

# The definitive Jupyter notebook. We'll try to do everything in here: interpolation, integration, and fomulation of the weight map. Let's go

# In[1]:


# Import Necessary Packages:
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits


# In[2]:


# We now need to plot the the dn/ds -Log(S) relationship. From Dai et al. 2015: 
a = 1.34
b = 2.37 # +/- 0.01
f_b = 3.67 * 10 ** (-15) # erg  cm^-2 s^-1
k = 531.91*10**14 # +/- 250.04; (deg^-2 (erg cm^-2 s^-1)^-1)
s_ref = 10**-14 # erg cm^-2 s^-1


# In[3]:


# We'll also try plotting some reference points from the paper:
logs_sample = [-14.58,-14.34,-14.10,-13.86,-13.62,-13.38,-13.14,-12.91,-12.67,-12.43,-12.19,-11.95,-11.71,-11.23]
dnds_sample = [2.26e17,5.37e16,2.03e16,8.05e15,2.53e15,6.29e14,1.53e14,3.83e13,8.32e12,1.99e12,7.28e11,1.03e11,2.96e10,7.32e09]

for i in range(0,len(logs_sample)):
    logs_sample[i] = 10**(logs_sample[i])


# In[4]:


# Input the broken power law for dn/ds from the paper:
def f1(x):
    return k*((x/s_ref)**-a)


def f2(x):
    return k*((f_b/s_ref)**(b-a))*(x/s_ref)**-b


# In[5]:


'''
plt.figure(figsize = [9, 8])

x1 = np.linspace(10**-18,f_b,10000)
x2 = np.linspace(f_b,10**-11, 10000)

y = np.linspace(0,f1(f_b),10)
x3 = []
for i in y:
    x3.append(f_b)




plt.plot(x1,f1(x1),color ='steelblue', label = 'Power-law fit')
plt.plot(x2,f2(x2), color = 'steelblue')
plt.plot(x3,y,'--') # Plot the break flux line as reference

plt.xscale("log")
plt.yscale("log")
plt.xlabel('Flux (erg cm^-2 s^-1)')
plt.ylabel('dN/ds (deg ^-2)')

plt.scatter(logs_sample,dnds_sample, color = 'indianred', marker = '^', label = 'Dai et. al 2015')
plt.legend()
#plt.savefig('dnds_test.png')
'''

# In[6]:



# In[7]:


# Read in the catalog for interpolation:
exp, fluxlim = np.loadtxt('/Users/bbonine/research/ou/corr_func/data/fluxlimit.txt',skiprows = 1, unpack = True)
exp


# In[8]:


for i in range (0,len(exp)):
    exp[i] = 10**exp[i]
    
exp


# In[9]:


fluxlim


# In[10]:


# Interpolate the flux values:
func1 = InterpolatedUnivariateSpline(exp,fluxlim) # 'fill_value' allows us to extrapolate beyond the table
xnew = np.linspace(0,10**8, num = 10**7, endpoint = True)


# In[11]:


# Let's try plotting the interpolation:
'''plt.plot(exp,fluxlim, 'o', label = 'Data')
plt.plot (xnew, func1(xnew), color = 'orange', label = 'Interpolation')
plt.xlabel('Exposure Time (s)')
plt.ylabel('Flux Limit (erg/s/cm^2)')
plt.xscale("Log")
plt.yscale("Log")
plt.legend()


plt.show()
'''

# In[12]:


# Read in the exposure map
image_file = '/Users/bbonine/research/ou/corr_func/data/grb121128a_expo.fits'

hdu_list = fits.open(image_file)
image_data = hdu_list[0].data
hdu_list.close()


# In[13]:


# Save reference pixel value for later
ref_flux =  image_data[500,500]
ref_flux


# In[14]:


# Use the interpolated function to extract flux limit based off reference flux
flux_lim = np.asscalar(func1(ref_flux))
flux_lim


# In[15]:


#Conver image data to 1D array:
exp_map_1d =  image_data.ravel()


# Find the flux limit for each pixel:
fluxlimit = []

for i in range(0,len(exp_map_1d)):
    fluxlimit.append(np.asscalar(func1(exp_map_1d[i])))


# So it looks like we have an array 'fluxlimit' with the corresponding flux limit of each pixel. Note that the lowest values in the exposure map have the largest flux limit (you need more flux with lower exposure)

# In[16]:


np.max(fluxlimit),np.min(fluxlimit)


# In[17]:


fluxlimit_1d = np.asarray(fluxlimit)
fluxlimit_2d = np.reshape(fluxlimit_1d,(-1,len(image_data[0])))


# Let's try making a visual representation of the exposure map using our flux limit conversion:

# In[18]:


# Save data as numpy array, then convert back into 2d:

fluxlimit_2d = np.reshape(fluxlimit_1d,(-1,len(image_data[0])))

plt.style.use('default')

plt.figure(figsize = [20, 20])
plt.imshow(fluxlimit_2d,cmap = 'binary', interpolation = 'none')

plt.colorbar(shrink = 1 ,orientation = 'horizontal')
plt.title('Exposure map: Flux limit per pixel')

#plt.savefig('expmap.png')


# Let's plot dn/ds for this data and compare it to the paper, then integrate to get N:

# In[19]:


dNpix = []

for i in range(0,len(fluxlimit_1d)):
    if fluxlimit_1d[i] <= f_b:
        dNpix.append(f1(fluxlimit_1d[i]))
    else :
        dNpix.append(f2(fluxlimit_1d[i]))


# In[20]:


np.max(dNpix),np.min(dNpix)


# In[21]:

'''
plt.style.use('default')
plt.figure(figsize = [9, 8])




#y = np.linspace(0,f1(f_b),10)
#x_break = []
#for i in y:
    #x3.append(f_b)




plt.plot(x1,f1(x1),color ='steelblue', label = 'Power-law fit')
plt.plot(x2,f2(x2), color = 'steelblue')


plt.xscale("log")
plt.yscale("log")
plt.xlabel('Flux (erg cm^-2 s^-1)')
plt.ylabel('dN/ds (deg ^-2)')

# Plot every 100th point
n = 100
plt.scatter(fluxlimit_1d[::n],dNpix[::n], color = 'indianred', marker = '^', label = 'Swift Exposure Map')
plt.plot(x3,y, color = 'black', linestyle = '--')  # Plot the break flux line as reference
plt.legend()

#plt.savefig('dnds_swift.png')'''


# Very cool. Let's proceed with the integration:

# In[22]:


# Integrate the broken power law from the paper:

def f3(x):
    return ((1/s_ref)**-a)*k*(1/(-a+1))*((f_b**(-a+1))-x**(-a+1)) 



def f4(x):
    return ((1/s_ref)**-b)*k*((f_b/s_ref)**(b-a))*(-x**(-b+1))


# In[23]:


# Now, we try to generalize this:

Npix = []

for i in range(0,len(fluxlimit_1d)):
    # check which range of the power law applies:
    if fluxlimit_1d[i] <= f_b:
        Npix.append(f3(fluxlimit_1d[i]) + f4(fluxlimit_1d[i]))
    
    else:
        Npix.append(f4(fluxlimit_1d[i]))


# In[24]:


N = np.abs(Npix)


# In[25]:


'''
plt.style.use('default')
plt.figure(figsize = [8, 8])

plt.xscale("Log")
plt.yscale("Log")


plt.ylabel("N / deg^2")
plt.xlabel("Flux limit (erg/s/cm^-2)")

# Plot every 100th point
q = 100
plt.scatter(fluxlimit_1d[::q],N[::q] ,color = 'indianred', marker = '^', label = 'Swift Exposure Map')
plt.legend()


#plt.savefig('N_swift.png')'''


# We're now ready to make the weight map. We start by calculating the number of sources in each pixel. 
# For Swift, the angular extent of 20 pixels is 47.1262 arcseconds. That means that the solid angle per pixel is 
#     
#         (47.1262" / 20)^2

# In[26]:


pixel_angle_sec = (47.1262 / 20)**2 # [square arcseconds]
pixel_angle_sec


# In[27]:


# Convert to square degrees: 1 deg = 3600"
pixel_angle_deg = (pixel_angle_sec / 3600**2)
pixel_angle_deg


# Mutliply the value of N /deg ^2 for each pixel by this conversion factor to get the number of sources per pixel:

# In[28]:


N_pixel = pixel_angle_deg*N
np.sum(N_pixel)


# Summing up the number of sources in each pixel over the image gives a value of 50.1. Approximately 50 sources in our mock image?

# In[29]:


np.max(N_pixel)


# The maximum number of sources in each pixel is 0.00225. We'll divide each pixel by this to get a normalized weight map:

# In[30]:


N_norm = N_pixel / np.max(N_pixel)
N_norm


# In[31]:


np.sum(N_norm),np.min(N_norm)


# Let's create a visual representation of the weight map:
# 

# In[32]:


# Save data as numpy array, then convert back into 2d:

weight_map = np.reshape(N_norm,(-1,len(image_data[0])))

plt.style.use('default')

plt.figure(figsize = [10, 10])
plt.imshow(weight_map,cmap = 'gray', interpolation = 'none')

plt.colorbar()
plt.title('Weight Map: Normalized sources  per pixel')


#plt.savefig('expmap.png')


# From here, I'll be adapting some old code I used to make a sample weight map:

# In[33]:


# Find the total weight
weight_tot = np.sum(N_norm)
weight_tot


# In[34]:


# Define an array for the 'edges' of each pixel in the map
weight_outer = np.cumsum(N_norm)
weight_outer


# The last entry is the total weight of the map, so this looks good. "weight_outer" sets the outer bound of the weight range for each pixel in the image. We now define the inner range as the outer range minus the weight of each pixel:

# In[35]:


weight_inner = weight_outer - N_norm
weight_inner




# In[36]:
# Read in data:
cat = '/Users/bbonine/research/ou/corr_func/data/agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
x,y = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(16,17) , unpack = True)

here = np.where(field == 'grb121128a')
data_x = x[here]
data_y = y[here]




# In[37]:


# Plot the weighted as a 2d image:



# We'll use this as our 'data'. Let's make another random image to use as the 'random' sample:

# In[38]:


n_sources = int(np.sum(N_pixel))
n_dim = 1000 # specify the dimmension of our image
img2 = np.zeros(n_dim*n_dim)
var = np.random.uniform(0,weight_tot,n_sources)
for j in range(0,n_sources):
    for i in range(0,len(img2)):
        if var[j] > weight_inner[i] and var[j] < weight_outer[i]:
            img2[i] = img2[i] + 1 # specifies flux of pixel. Add one 'photon' if 


# In[39]:


# Plot the weighted as a 2d image:
plt.figure(figsize = [15, 15])
rand_img = np.reshape(img2,(n_dim,n_dim)) 
plt.imshow(rand_img,cmap = 'gray')
plt.colorbar()
plt.show()









# In[42]:


# Visulization of the 'data'
plt.style.use('dark_background')
plt.figure(figsize = [12, 12])

plt.scatter(data_x,data_y, marker = '.', color = 'white', s = 25)
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.show()


# In[43]:


here2 = np.where(rand_img > 0)

rand_x = here2[0] # image position of x values
rand_y = here2[1] # image position of y vales 


# In[44]:


# Visulization of the random image
plt.figure(figsize = [12, 12])
plt.scatter(rand_x,rand_y, marker = '.', color = 'white', s = 25)
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.show()


# In[45]:


# Keep track of each pixel's index (probably extraneous step)
#x_index=[]
#y_index=[]
#for i in range(0,len(rand_x)): 
    #x_index.append(i)
    #y_index.append(i)

# Find the unique permutations between indices:
#perm = np.asarray(list(itertools.permutations(x_index,2)))
    


# To get the distance between two pixels, we need evaluate x(perm[i,0]) and x(perm[i,1]) and subtract x values. We do the same for y and iterate through all i entries in the permutation array. From there, we do just do Pythagorean theorem 

# In[46]:


# Calculate the separation between 'sources' in the random image
#dx_rand = []
#dy_rand = []
#for i in range(0,len(perm)):
    #dx_rand.append(rand_x[perm[i,1]] - rand_x[perm[i,0]])
    #dy_rand.append(rand_y[perm[i,1]] - rand_y[perm[i,0]])


# In[47]:


# New approach: Iterate through the image to get combinations.

# Define pixel distance function between two sources:
def distance(x2,x1,y2,y1):
    return (((x2-x1)**2 + (y2-y1)**2)**0.5)


# In[48]:


# Test the distance function:
distance(10,5,20,17)


# In[49]:


# Iterate through the random points:
dist_rr = []
for i in range(len(rand_x)):
    for j in range(len(rand_x)):
        if j != i:
            dist_rr.append(distance(rand_x[j],rand_x[i],rand_y[j],rand_y[i]))


# In[50]:


np.max(dist_rr),np.min(dist_rr)


# We now have the pixel distance between all pairs of points in the random image. Now, recall that the SWIFT XRT has an angular scale of (47.1262" / 20 pixels), or  

# In[51]:


pix_scale = 47.1262 / 20 # arcseconds / pixel
pix_scale


# In[52]:


# Convert pixel distance to angular distance:

rr_ang_dist = pix_scale * np.asarray(dist_rr) # arcsec/pix * pix = arcsec
rr_ang_dist 


# In[53]:


# Repeat same process for data-data:
dist_dd = []
for i in range(len(data_x)):
    for j in range(len(data_x)):
        if j != i:
            dist_dd.append(distance(data_x[j],data_x[i],data_y[j],data_y[i]))


# In[54]:


# Convert DD pair separation to arcseconds:
dd_ang_dist = pix_scale * np.asarray(dist_dd)
dd_ang_dist


# In[55]:


# And again for the data-random pairs:
dist_dr = []
for i in range(len(data_x)):
    for j in range(len(rand_x)):
        if j != i:
            dist_dr.append(distance(rand_x[j],data_x[i],rand_y[j],data_y[i]))


# In[56]:


# And convert DR pair separation to arcseconds:
dr_ang_dist = pix_scale * np.asarray(dist_dr)
dr_ang_dist


# In[57]:


np.max(dr_ang_dist),np.max(dd_ang_dist),np.max(rr_ang_dist)


# We now have arrays for the angular distance between data-data, data-random, and random-random pairs. Next, we need to bin the data:

# In[69]:


# Bin the data automatically for one of them:
dd_binned = np.histogram(dd_ang_dist)

dd_binned


# In[64]:


# choose one of the binning schemes:

bins = dd_binned[1] # selects the 'bins' array from np.histogram
bins


# In[95]:


# Select the angular separations
dd = np.histogram(dd_ang_dist, bins = bins)[0]
dr = np.histogram(dr_ang_dist, bins = bins)[0]
rr = np.histogram(rr_ang_dist, bins = bins)[0]

len(dd)


# In[104]:


N_d = len(data_x)
N_r = len(rand_x)

N = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))
N


# In[140]:


# Now, let's try it for each bin:
#N = ((len(data_x)*len(rand_x))**2) / (len(data_x)(len(data_x)-1)*(len(rand_x)(len(rand_x)-1)))
def W(DD,DR,RR):
    return  (N *((DD * RR) / (DR)**2) ) -1


# In[141]:


corr = []
for i in range(0,len(dr)):
    corr.append(W(dd[i],dr[i],rr[i]))
    


# In[142]:


corr


# In[139]:



plt.style.use('default')
plt.figure(figsize = [12, 8])
plt.plot(bins[0:10],corr[0:],'--')
plt.xlabel('Angular Separation (Arcseconds)')
plt.xscale('log')
plt.ylabel(r'W$(\theta)$')
plt.show()


# In[143]:


# Now, we need to calculate the varience: 
# sigma^2 = 3 [(1+W)^2] / DD

varr = []
for i in range(0,len(dr)):
    varr.append(3*(1+(W(dd[i],dr[i],rr[i])))**2 / dd[i])


# In[144]:


len(varr),len(corr)


# In[148]:





# In[150]:



plt.style.use('default')
plt.figure(figsize = [12, 8])


plt.errorbar(bins[0:10],corr, yerr = np.sqrt(varr), fmt = '.')
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.show()


# In[ ]:




