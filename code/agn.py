# Preform correlation function computation on actual agn data:
# Import Necessary Packages:
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.neighbors import KDTree
import os



# Read in AGN table:
path = '/Users/bbonine/ou/research/corr_func/data/'
# Remote version: cat = "/home/bonine/donnajean/research/agn_corr/data/agntable_total.txt"
cat = path + 'agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
x,y = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(16,17) , unpack = True)


# Read in the flux limit file: 
#lim = '/home/bonine/donnajean/research/agn_corr/data/fluxlimit.txt'
lim = path + 'fluxlimit.txt'
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

path1 = "/Users/bbonine/ou/research/corr_func/outputs_3/"
path3 = "/Users/bbonine/ou/research/corr_func/figures/12_9/"
os.mkdir(path1)


#######################################
# Hamilton Estimator
def W_ham(N,DD,DR,RR):
    return (N *(( DD* RR) / (DR)**2) ) -1

# Landy- Salazay Estimator
def W_ls(DD,DR,RR):
    return (( DD - (2*DR) + RR) / (RR)) -1


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin Looping through each exposure map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''



# Make null arrays for pair counts
# Specify binning
num_bins = 8
bins = np.logspace(0,2.5, num_bins)
bins = np.linspace(0,600,num_bins) # pixels 



# Total number of data and random points (append in loop)
N_d = 0
N_r = 0
field_count = 0



#Begin looping through fields

# Swift Params
pixel_angle_sec = (47.1262 / 20)**2 # [square arcseconds]
pixel_angle_deg = (pixel_angle_sec / 3600**2)
pix_scale = 47.1262 / 20 # arcseconds / pixel

# From Dai et al 2015::
a = 1.34
b = 2.37 # +/- 0.01
f_b = 3.67 * 10 ** (-15) # erg  cm^-2 s^-1
k = 531.91*10**14 # +/- 250.04; (deg^-2 (erg cm^-2 s^-1)^-1)
s_ref = 10**-14 # erg cm^-2 s^-1



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

################################################
# Specify number of fields to include here:

#loops = np.len(field_list) # all fields
loops = 30

# Null array to populate with correlation function values
# Note the shape; 
corr = np.zeros((loops,len(bins)-1))
varr = np.zeros((loops,len(bins)-1))
ratio = np.zeros(loops)
flag = np.zeros(loops)


# Begin loop
for i in range(0,loops):
    # Swift telescope values from Dai et al. (2015)
    a = 1.34
    b = 2.37 # +/- 0.01
    f_b = 3.67 * 10 ** (-15) # erg  cm^-2 s^-1
    k = 531.91*10**14 # +/- 250.04; (deg^-2 (erg cm^-2 s^-1)^-1)
    s_ref = 10**-14 # erg cm^-2 s^-1

    def f3(x):
        return (1/(-a+1))*(1/s_ref)**(-a)*k*(x**(-a+1))

    def f4(x):
        return (1/s_ref)**(-b)*k*(f_b/s_ref)**(b-a)*(1/(-b+1))*(x**(-b+1))
    

    
    # Read in the relevant exposure map:
    here = np.where(field == field_list[i])
    # Extract source positions in this field:
    #data_x = x[here]
    #data_y = y[here]
     
    # Check for exposure map 
    if os.path.isfile(path  + field_list[i] +'/expo.fits') == True:
        expmap = path + field_list[i] +'/expo.fits'
        print("Exposure map located")
        # Make directory for outuput files for this field:
        path2 = path1+field[here][0]
        os.mkdir(path2)
        print("Directory created...")
        
        field_count += 1
        
        # Read in exposure map with astropy
        hdu_list = fits.open(expmap)
        image_data = hdu_list[0].data
        hdu_list.close()
        exp_map_1d =  image_data.ravel() #Conver exposure map to 1D array for later
        
        # Restrict to fields with more than one AGN (necessary for correlation calculation):
    
        # Save reference pixel value for later
        ref_flux =  image_data[500,500]
    
        # Use the interpolated function to extract flux limit based off reference flux
        flux_lim = func1(ref_flux)
    
        # Find the flux limit for each pixel:
        fluxlimit = np.zeros(len(exp_map_1d))
        for j in range(0,len(fluxlimit)):
            fluxlimit[j] = func1(exp_map_1d[j])
            
        fluxlimit_1d = np.asarray(fluxlimit) #convert to numpy array
        fluxlimit_2d = np.reshape(fluxlimit_1d,(-1,len(image_data[0])))
    
        # Determine number of sources per pixel
        Npix = []
        for j in range(0,len(fluxlimit_1d)):
            if fluxlimit_1d[j] <= f_b:
                Npix.append(f3(fluxlimit_1d[j]))
            else:
                Npix.append(f4(fluxlimit_1d[j]))
    
        N = np.abs(Npix)
        N_source = pixel_angle_deg*N # Number of sources
        N_norm = N_source / np.max(N_source) # Normalize
    
        # Construct weight map to gerenate random image:
        weight_map = np.reshape(N_norm,(-1,len(image_data[0])))
        plt.style.use('default')
        plt.figure(figsize = [10, 10])
        plt.imshow(weight_map,cmap = 'gray', interpolation = 'none', origin = 'lower')
        plt.colorbar()
        plt.title('Field '+field[here][0]+ ': Normalized sources per pixel')
        plt.savefig(path2+'/expmap.png')
        plt.close()
        print("Exposure map " + str(i+1) + " created..." )
        
        
    
        # Begin making random image:
        weight_tot = np.sum(N_norm) 
        weight_outer = np.cumsum(N_norm) # 'Outer edge' of pixel weight
        weight_inner = weight_outer - N_norm # 'Inner edge' of pixel weight
        n_sources = int(np.sum(N_source)) 
        n_dim = 1000 # specify the dimmension of our image
        img2 = np.zeros(n_dim*n_dim)
        #img3 = np.zeros(n_dim*n_dim) # delete this if only using one random image; added 11/15
        var1 = np.random.uniform(0,weight_tot,n_sources)
        var2 = np.random.uniform(0,weight_tot,n_sources)
        for l in range(0,n_sources):
            for m in range(0,len(img2)):
                if var1[l] > weight_inner[m] and var1[l] < weight_outer[m]:
                    img2[m] = img2[m] + 1 # specifies flux of pixel. 
    
        # Save random image to file:
        rand_img = np.reshape(img2,(n_dim,n_dim)) 
        here2 = np.where(rand_img > 0)
        rand_x = here2[0] # image position of x values
        rand_y = here2[1] # image position of y vales
        N_r = len(rand_x) # Tally number of random pionts
        
        
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Test 11/14: Repeat random image generation; call this one the data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for l in range(0,n_sources):
            for m in range(0,len(img3)):
                if var2[l] > weight_inner[m] and var2[l] < weight_outer[m]:
                    img3[m] = img3[m] + 1 # specifies flux of pixel. 
         '''
        # Save random image to file:
        #data_img2 = np.reshape(img3,(n_dim,n_dim)) 
        #here3 = np.where(data_img2 > 0)
        data_x = x[here] # image position of x values
        data_y = y[here] # image position of y vales
        N_d = len(data_x) # Tally number of points
        
        
        
        plt.style.use('dark_background')
        plt.figure(figsize = [12, 12])
        plt.scatter(rand_x,rand_y, marker = '.', color = 'white', s = 25)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.title('Field '+field[here][0]+ ': Random Image')
        plt.savefig(path2 + '/rand_img.png')
        plt.close()
        # Save data image to file:
        plt.style.use('dark_background')
        plt.figure(figsize = [12, 12])
        plt.scatter(data_x,data_y, marker = '.', color = 'white', s = 25)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.title('Field '+field[here][0]+ ': Data Image')
        plt.savefig(path2 + '/data_img.png')
        plt.close()
        print("Random Image" + str(i+1) + " created...")
            
            

        # Begin calculating correlation function
        
        '''
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           12/8: Try KD tree implementation to get pair counts
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         '''
        if len(data_x) and len(rand_x) > 0:
           
            ##########################################
            # Format data for sklearn
            data = np.vstack((data_x,data_y)).T
            data_rand = np.vstack((rand_x,rand_y)).T
           
            #############################################################
            # Set up KD trees
            KDT_D = KDTree(data)
            KDT_R = KDTree(data_rand)
            
                       
            #############################################################
            # Tally pair counts
            counts_DD = KDT_D.two_point_correlation(data, bins)
            counts_RR = KDT_R.two_point_correlation(data_rand, bins)
            counts_DR = KDT_R.two_point_correlation(data, bins)

            
            DD = np.diff(counts_DD)
            RR = np.diff(counts_RR)
            DR = np.diff(counts_DR)

            
            # Determine ratio of random-to-data pairs
            ratio[i] = len(rand_x) / len(data_x)
        
           
            ###################################################
            #Compute Correlation function; 
            factor = len(data_rand) * 1. / len(data)
            corr[i] = W_ham(factor,DD,DR,RR)
            
            
            # Varience
        
            varr[i] = 3*((1+(corr[i])**2) / DD)
        

          


# Flag fields      
nan_check_corr = np.isnan(corr)
inf_check_varr = np.isinf(varr)
for i in range(0,len(corr)):
    if 1 in nan_check_corr[i] or 1 in inf_check_varr[i]:       # Check for NaN, inf
        flag[i] = 1
    
    if ratio[i] > 3:                                           # Check for high ratios
        flag[i] = 1                         
# Select unflagged fields  
here4 = np.where(flag == 0)[0]    

corr = corr[here4]
varr = varr[here4]    
       
  

corr_mean = np.mean(corr, axis = 0)    
varr_mean = np.mean(varr, axis = 0)

corr_weight = np.average(corr,axis = 0, weights = varr)      

#Plot 
centers = 0.5*(bins[1:]+ bins[:-1])*pix_scale

plt.style.use('default')
plt.figure(figsize = [12, 8], dpi = 300)

plt.errorbar(centers,corr_mean, yerr = np.sqrt(varr_mean), fmt = '.', capsize = 5, label = "Fields 1 - 20", marker = 's', color = 'maroon', ms = 4)
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('Mean Correlation Function')
plt.savefig(path3 + 'mean.png')
plt.close


plt.figure(figsize = [12, 8], dpi = 300)

plt.errorbar(centers,corr_weight, yerr = np.sqrt(varr_mean), fmt = '.', capsize = 5, label = "Fields 1 - 20", marker = 's', color = 'maroon', ms = 4)
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('Weighted Average Correlation Function')
plt.savefig(path3+'weight_avg.png')
plt.close










''' 
###################################################     
# Begin calculating stacked correlation function
    
N_norm= (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))



# Landy- Salazay Estimator
def W_ls(DD,DR,RR):
    return (( DD - (2*DR) + RR) / (RR)) -1


corr_ham = W_ham(dd_stack,dr_stack,rr_stack)


#Varience:
varr = 3*((1+(corr_ham)**2) / dd_stack)




# Save output arrays to file
np.savetxt(path1+ '/out_1.txt', (centers[:],corr_ham,varr), delimiter = ',')
np.savetxt(path1+ '/out_2.txt', (centers[:],corr_ls,varr), delimiter = ',')
np.savetxt(path1+'/ratios.txt', (ratio), delimiter = ',')
print("Correlation Analysis complete. Have a great day!")

centers = 0.5*(bins[1:]+ bins[:-1])
# Plot results
plt.style.use('default')
plt.figure(figsize = [12, 8])
plt.errorbar(centers,corr_5, yerr = np.sqrt(varr_5), fmt = '.', capsize = 5, label = "Fields 1 - 50", marker = 's', color = 'maroon', ms = 4)

plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('Stacked Correlation Function')
plt.xscale('log')
plt.yscale('log')
plt.savefig(path1 + '/log_corr.png')
plt.close()corr




        

        

plt.style.use('default')
plt.figure(figsize = [12, 8])
plt.errorbar(bins[0:10],corr_1, yerr = np.sqrt(varr_1), fmt = '.', capsize = 5, label = "Fields 1 - 184", marker = 's', color = 'black', ms = 4)
plt.errorbar(bins[0:10],corr_2, yerr = np.sqrt(varr_2), fmt = '.', capsize = 5, label = "Fields 185 - 369", marker = '^', color = 'darkgray', ms = 4)
plt.errorbar(bins[0:10],corr_3, yerr = np.sqrt(varr_3), fmt = '.', capsize = 5, label = "Fields 370 - 551", marker = 's', color = 'cornflowerblue', ms = 4)
plt.errorbar(bins[0:10],corr_4, yerr = np.sqrt(varr_4), fmt = '.', capsize = 5, label = "Fields 552 - 739", marker = 's', color = 'maroon', ms = 4)
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('Stacked Correlation Function')
plt.legend()      
    
x = np.linspace(0,1000, num = 100)
plt.plot(x,x**(-1.4), label = r'$\gamma = -1$')
plt.errorbar(bins[0:10],corr_1, yerr = np.sqrt(varr_1), fmt = '.', capsize = 5, label = "Fields 1 - 184", marker = 's', color = 'black', ms = 4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.legend()
'''