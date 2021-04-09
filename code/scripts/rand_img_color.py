'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Filename: rand_img_color.py
Actions:
    - Calcualte color excess for each AGN
    - generate random catalogs for red and blue samples
    - Plot and save outputs
Author: Brett Bonine
bonine@ou.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
print("-----------------------------------------------------------------------")
print("rand_img_color.py: Generate Random Images using MIR colors")
print("-----------------------------------------------------------------------")

######################################################################
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os       # make outputs dirs

from astropy.io import fits # read in exposure map data
from scipy.interpolate import InterpolatedUnivariateSpline # interpolate flux vals


# Default tick label size

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin Main Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#######################################################################
# Read in field list and fluxlimit info
#path_fields = '/Users/bbonine/ou/research/corr_func/data/'

# Remote path:




path_fields = '/Users/bonine/research/corr_func/data/'
cat = path_fields + 'agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
w1,w2,x,y = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(10,11,16,17) , unpack = True)
n_agn = len(x)

field_list = np.unique(field)  # clear duplicates

# Convert to strings to upper case (necessary to match exposure map directory names on remote)
for i in range(0,len(field_list)):
    field_list[i] = field_list[i].upper()




lim = path_fields + 'fluxlimit.txt'
exp, fluxlim = np.loadtxt(lim,skiprows = 1, unpack = True)
exp = np.power(10,exp)  #exposure time in log units; convert

# Interpolate the flux values:
func1 = InterpolatedUnivariateSpline(exp,fluxlim)
xnew = np.linspace(0,10**8, num = 10**7, endpoint = True)

########################################################################
# Sort by color
color_flag = np.zeros(n_agn)
w_mag = np.zeros(n_agn)

'''
Flag guide:
0: MIR Red [w1-w2 > 0.35 mag] (Unobscured)
1: MIR Blue [w1-m2 < 0.35 mag] (Obscured)
'''
for i in range(0,n_agn):
    w_mag[i] = w1[i] - w2[i]
    if w_mag[i] < 0.35:
        color_flag[i] = 1

# Determine ratio of sources 
red_cnts = len(x[color_flag == 0])
blue_cnts = len(x[color_flag == 1])

red_frac = red_cnts/n_agn
blue_frac = blue_cnts/n_agn



###############################
# SWIFT XRT params
# Swift Params
pixel_angle_sec = (47.1262 / 20)**2 # [square arcseconds]
pixel_angle_deg = (pixel_angle_sec / 3600**2)
pix_scale = 47.1262 / 20 # arcseconds / pixel

#################################################################
# Output Arrays
loops = len(field_list) # all fields

exp_mean = np.zeros(loops)

#####################################################################
# Setup output folders
out = input("Please Enter Output Directory Name: ")
#path_out = "/Users/bbonine/ou/research/corr_func/outputs/color/" + out +"/"
# Remote path:
path_out = "/Users/bonine/research/corr_func/outputs/color/"+out + "/"
print("Outputs will be saved to " + path_out)
if os.path.isdir(path_out) == False:
    os.mkdir(path_out)

f = open(path_out+ "random_cat.txt", "a")
f.write("GRB\tIMG_X\tIMG_Y\tFLAG\n")

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Begin main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
print("-----------------------------------------------------------------------")
print("Beginning Main Loop....")
for i in range(0,loops):
    # Swift telescope values from Dai et al. (2015)
    a = 1.34
    b = 2.37 # +/- 0.01
    f_b = 3.67 * 10 ** (-15) # erg  cm^-2 s^-1
    k = 531.91*10**14 # +/- 250.04; (deg^-2 (erg cm^-2 s^-1)^-1)
    s_ref = 10**-14 # erg cm^-2 s^-1

# Analytically integrate above dn/ds relations using appropriate bounds
    def n_1(s):
        return k*(1/(-a+1))*(1/s_ref)**(-a)*(f_b**(-a+1)-s**(-a+1))

    def n_2(s):
        return k*(f_b/s_ref)**(b-a)*(1/s_ref)**(-b)*(1/(-b+1))*(-s**(-b+1))



    # Read in the relevant exposure map:
    target = np.where(field == field_list[i])[0]
           
    # Values associated with this field
    data_x = x[target]
    data_y = y[target] 
    flags = color_flag[target]


    # Check for exposure map
    path_exp = '/Volumes/hd5/swift/grb/'
    if os.path.isfile(path_exp  + field_list[i] +'/expo.fits') == True:
        expmap = path_fields + field_list[i] +'/expo.fits'
        
        # Make directory for outuput files for this field:
        path2 = path_out+field[target][0]
        if os.path.isdir(path2) == False:
            os.mkdir(path2)
        

        # Read in exposure map with astropy
        hdu_list = fits.open(expmap)
        image_data = hdu_list[0].data
        hdu_list.close()
        exp_map_1d =  image_data.ravel() #Convert exposure map to 1D array for later

        # Record mean value
        exp_mean[i] = np.mean(exp_map_1d)

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
        Npix = np.zeros(len(fluxlimit_1d))


        # Integrate dn/ds to find number of AGN per square degree for given flux limits
        for j in range(0,len(fluxlimit_1d)):
            if fluxlimit_1d[j] > f_b:
                Npix[j] = n_2(fluxlimit_1d[j])

            else:
                Npix[j] = n_1(fluxlimit_1d[j]) +n_2(f_b)


        N = np.abs(Npix)
        N_source = pixel_angle_deg*N # Number of sources per square arcsecond
        N_norm = N_source / np.max(N_source) # Normalize

        # Construct weight map to gerenate random image:
        weight_map = np.reshape(N_norm,(-1,len(image_data[0])))

        plt.style.use('default')
        plt.figure(figsize = [8, 8])
        plt.imshow(weight_map,cmap = 'gray', interpolation = 'none', origin = 'lower', norm = LogNorm())
        plt.title('Field '+field[target][0]+ ': Normalized sources per pixel')
        #plt.savefig(path2+'/expmap.png')
        plt.close()
        print("Exposure map " + str(i+1) + " created..." )



        # Begin making random image:
        weight_tot = np.sum(N_norm)
        weight_outer = np.cumsum(N_norm) # 'Outer edge' of pixel weight
        weight_inner = weight_outer - N_norm # 'Inner edge' of pixel weight

        # Determine total number of AGN in field
        n_sources = int(np.sum(N_source))

        # Segregate by color
        n_red = int(10*red_frac*n_sources)
        n_blue = int(10*blue_frac*n_sources)

        n_dim = 1000 # specify the dimmension of our image
        rand_img_red = np.zeros(n_dim*n_dim)
        rand_img_blue = np.zeros(n_dim*n_dim) # delete this if only using one random image; added 11/15

        # Draw random weight map values
        var_red = np.random.uniform(0,weight_tot,n_red)
        var_blue = np.random.uniform(0,weight_tot,n_blue)

        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Construct random images:
         Iterate through weight map; compare value of random draw
         to associated weight map boundaries for each pixel.
         If draw falls within boundaries, update pixel value
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        '''
        ########################################
        # Red: 
        for l in range(0,n_red):
            for m in range(0,len(rand_img_red)):
                if (var_red[l] > weight_inner[m]) and (var_red[l] < weight_outer[m]):
                    rand_img_red[m] = rand_img_red[m] + 1 # specifies flux of pixel.

        # Save random image to file:
        rand_img_red = np.reshape(rand_img_red,(n_dim,n_dim))
        rand_srcs_red = np.where(rand_img_red > 0)

        # Extract data and random points 
        rand_x_red = rand_srcs_red[1] # image position of x values
        rand_y_red = rand_srcs_red[0] # image position of y vales

        data_x_red = data_x[flags ==0]
        data_y_red = data_y[flags ==0]


        ########################################
        # Blue: 
        for l in range(0,n_blue):
            for m in range(0,len(rand_img_blue)):
                if (var_blue[l] > weight_inner[m]) and (var_blue[l] < weight_outer[m]):
                    rand_img_blue[m] = rand_img_blue[m] + 1 # specifies flux of pixel.

        # Save random image to file:
        rand_img_blue = np.reshape(rand_img_blue,(n_dim,n_dim))
        rand_srcs_blue = np.where(rand_img_blue > 0)

        # Extract data and random points 
        rand_x_blue = rand_srcs_blue[1] # image position of x values
        rand_y_blue = rand_srcs_blue[0] # image position of y vales

        data_x_blue = data_x[flags ==1]
        data_y_blue = data_y[flags ==1]
        # Save data to file
        for j in range(0,len(rand_x_red)):
            f.write(field[target][0]+"\t"+repr(rand_x_red[j])+"\t"+repr(rand_y_red[j])+"\t" +repr(0)+"\n")
        for j in range(0,len(rand_x_blue)):
            f.write(field[target][0]+"\t"+repr(rand_x_blue[j])+"\t"+repr(rand_y_blue[j])+"\t" +repr(1)+"\n")
        print("Random sources for field "+ str(i+1) +" generated. ")
f.close()
print("----------------------------------------- SUMMARY -----------------------------------------------------")
print("Random Catalog Sucessfully created at " + path_out+ "random_cat.txt")
print(" ")
print("Have a nice day!")
print("------------------------------------------------------------------------------------------------------")