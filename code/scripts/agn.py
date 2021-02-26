# Preform correlation function computation on actual agn data:
# Import Necessary Packages:
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
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


# Get rid of duplicates
field_list = np.unique(field)

# Create output folder

path1 = "/Users/bbonine/ou/research/corr_func/outputs_test_12_22_2/"
path3 = "/Users/bbonine/ou/research/corr_func/figures/12_22_2/"
os.mkdir(path1)
os.mkdir(path3)









# Total number of data and random points (append in loop)
N_d = 0
N_r = 0
field_count = 0





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





################################################
# Specify number of fields to include here:

#loops = len(field_list) # all fields
loops = 300 

# Make null arrays for pair counts
#bins = np.logspace(0,3.1,10) #logbins
num_bins = 10
bins = np.linspace(0,1200,num_bins)

dd_stack = np.zeros(num_bins - 1)
dr_stack = np.zeros(num_bins -1)
rr_stack = np.zeros(num_bins  -1)


# Total number of data and random points (append in loop)
N_d = 0
N_r = 0
field_count = 0

ratio = []

# Calculate distance between points
 
def distance(x2,x1,y2,y1):
    return (((x2-x1)**2 + (y2-y1)**2)**0.5)

# Hamilton orrelation function estimator:


def W_ham(DD,DR,RR):
    return (N*(( DD* RR) / (DR)**2) ) -1



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
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
        
        
        
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Test 11/14: Repeat random image generation; call this one the data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for l in range(0,n_sources):
            for m in range(0,len(img3)):
                if var2[l] > weight_inner[m] and var2[l] < weight_outer[m]:
                    img3[m] = img3[m] + 1 # specifies flux of pixel. 
         '''
        # And now the data:
        #data_img2 = np.reshape(img3,(n_dim,n_dim)) 
        #here3 = np.where(data_img2 > 0)
        
        
        data_x = x[here] # image position of x values
        data_y = y[here] # image position of y vales
        
        
        
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
           12/15: Revert to Euclidean method
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         '''
        if len(data_x) and len(rand_x) > 0:
           
            dist_rr = []
            for j in range(len(rand_x)):
                for k in range(len(rand_x)):
                    if k != j:
                        dist_rr.append(distance(rand_x[k],rand_x[j],rand_y[k],rand_y[j]))
                        

            
            # Repeat same process for data-data:
            dist_dd = []
            for j in range(len(data_x)):
                for k in range(len(data_x)):
                    if k != j:
                        dist_dd.append(distance(data_x[k],data_x[j],data_y[k],data_y[j]))
            
            # And data-random:
            dist_dr = []
            for j in range(len(data_x)):
                for k in range(len(rand_x)):
                    dist_dr.append(distance(rand_x[k],data_x[j],rand_y[k],data_y[j]))
            
            
            
            
            # Convert to angular distance
            rr_ang_dist = pix_scale * np.asarray(dist_rr) # arcsec/pix * pix = arcsec
            dd_ang_dist = pix_scale * np.asarray(dist_dd)
            dr_ang_dist = pix_scale * np.asarray(dist_dr)
            
            
            
            # Bin 
            dd = np.histogram(dd_ang_dist, bins = bins)[0]
            dr = np.histogram(dr_ang_dist, bins = bins)[0]
            rr = np.histogram(rr_ang_dist, bins = bins)[0]
            

        
            
            # Collect ratio of data to random points
            
            if ((len(data_x) / len(rand_x)) <= 2) and ((len(data_x) / len(rand_x)) >= 0.4):
                ratio.append(len(data_x) / len(rand_x))
                dd_stack += (dd/2 )
                dr_stack += dr /2
                rr_stack += (rr/2)
                        
                N_d += len(data_x)
                N_r += len(rand_x)
                field_count += 1

          


 # Evaluate   
N = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))

varr = 3*((1+(corr)**2) / dd_stack)

#Plot 
centers = 0.5*(bins[1:]+ bins[:-1])

print("correlation analysis complete! Have a great day. You're going to kill it.")
'''
plt.style.use('default')
plt.figure(figsize = [12, 8], dpi = 300)

plt.errorbar(centers,corr, yerr = np.sqrt(varr), fmt = '.', capsize = 5, label = "Fields 1 - 20", marker = 's', color = 'maroon', ms = 4)
plt.xlabel('Angular Separation (Arcseconds)')
plt.ylabel(r'W$(\theta)$')
plt.title('Correlation Function')
plt.savefig(path3 + 'mean.png')
plt.close'''












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