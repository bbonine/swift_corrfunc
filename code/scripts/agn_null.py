# Preform correlation function computation on actual agn data:
# Import Necessary Packages:
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

# Create output folders

path1 = "/Users/bbonine/ou/research/corr_func/outputs/02_15_21_rand_15bin/"
path3 = "/Users/bbonine/ou/research/corr_func/figures/02_15_21_rand_15bin/"
path4 = path3 + "flag/"

if os.path.isdir(path1) == False:
    os.mkdir(path1)


if os.path.isdir(path3) == False:
    os.mkdir(path3)


if os.path.isdir(path4) == False:
    os.mkdir(path4)








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



loops = len(field_list) # all fields


# Make null arrays for pair counts
#bins = np.logspace(0,3.1,10) #logbins
num_bins = 15
bins_lin = np.linspace(0,630,num_bins)
centers_lin = 0.5*(bins_lin[1:]+ bins_lin[:-1])



#####################################################
#Setup Null arrays to populate
corr_lin = np.zeros((loops,len(bins_lin)-1))
varr_lin = np.zeros((loops,len(bins_lin)-1))


flag = np.zeros(loops)


rand_counts = np.zeros(loops)
data_counts = np.zeros(loops)


exp_mean = np.zeros(loops)
num_source = np.zeros(loops)


field_count = 0

ratio = []

# Calculate distance between points

def distance(x2,x1,y2,y1):
    return (((x2-x1)**2 + (y2-y1)**2)**0.5)

# Hamilton orrelation function estimator:


def W_ham(N,DD,DR,RR):
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

    '''
    def f3(x):
        return (1/(-a+1))*(1/s_ref)**(-a)*k*(x**(-a+1))

    def f4(x):
        return (1/s_ref)**(-b)*k*(f_b/s_ref)**(b-a)*(1/(-b+1))*(x**(-b+1))
    '''


# Analytically integrate above dn/ds relations using appropriate bounds
    def n_1(s):
        return k*(1/(-a+1))*(1/s_ref)**(-a)*(f_b**(-a+1)-s**(-a+1))

    def n_2(s):
        return k*(f_b/s_ref)**(b-a)*(1/s_ref)**(-b)*(1/(-b+1))*(-s**(-b+1))



    # Read in the relevant exposure map:
    target = np.where(field == field_list[i])[0]
    # Extract source positions in this field:
    #data_x = x[here]
    #data_y = y[here]

    # Check for exposure map
    if os.path.isfile(path  + field_list[i] +'/expo.fits') == True:
        expmap = path + field_list[i] +'/expo.fits'
        print("Exposure map located")
        # Make directory for outuput files for this field:
        path2 = path1+field[target][0]
        os.mkdir(path2)
        print("Directory created...")



        # Read in exposure map with astropy
        hdu_list = fits.open(expmap)
        image_data = hdu_list[0].data
        hdu_list.close()
        exp_map_1d =  image_data.ravel() #Conver exposure map to 1D array for later

        # Record mean value
        exp_mean[i] = np.mean(exp_map_1d)

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
        Npix = np.zeros(len(fluxlimit_1d))


        # Integrate dn/ds to find number of AGN per square degree for given flux limits
        for j in range(0,len(fluxlimit_1d)):
            if fluxlimit_1d[j] > f_b:
                Npix[j] = n_2(fluxlimit_1d[j])

            else:
                Npix[j] = n_1(fluxlimit_1d[j]) +n_2(f_b)



        N = np.abs(Npix)
        N_source = pixel_angle_deg*N # Number of sources per sqiare arcsecond
        N_norm = N_source / np.max(N_source) # Normalize

        # Construct weight map to gerenate random image:
        weight_map = np.reshape(N_norm,(-1,len(image_data[0])))

        plt.style.use('default')
        plt.figure(figsize = [8, 8])
        plt.imshow(weight_map,cmap = 'gray', interpolation = 'none', origin = 'lower', norm = LogNorm())
        plt.title('Field '+field[target][0]+ ': Normalized sources per pixel')
        plt.savefig(path2+'/expmap.png')
        plt.close()
        print("Exposure map " + str(i+1) + " created..." )



        # Begin making random image:
        weight_tot = np.sum(N_norm)
        weight_outer = np.cumsum(N_norm) # 'Outer edge' of pixel weight
        weight_inner = weight_outer - N_norm # 'Inner edge' of pixel weight


        n_sources = int(np.sum(N_source))


        # Record number of sources
        #num_source[i] = n_sources

        n_dim = 1000 # specify the dimmension of our image
        img2 = np.zeros(n_dim*n_dim)
        img3 = np.zeros(n_dim*n_dim) # delete this if only using one random image; added 11/15

        # Draw random weight map values
        var_1 = np.random.uniform(0,weight_tot,n_sources)
        var_2 = np.random.uniform(0,weight_tot,n_sources) # Make a second random image




        '''
        %%%%%%%%%%%%%%%%%%%%%%%
        Construct random image
        %%%%%%%%%%%%%%%%%%%%%%%
        '''

       ###########################################################
        # Iterate through weight map; compare value of random draw
        # to associated weight map boundaries for each pixel.
        # If draw falls within boundaries, update pixel value
        for l in range(0,n_sources):
            for m in range(0,len(img2)):
                if (var_1[l] > weight_inner[m]) and (var_1[l] < weight_outer[m]):
                    img2[m] = img2[m] + 1 # specifies flux of pixel.

        # Save random image to file:
        rand_img = np.reshape(img2,(n_dim,n_dim))
        here2 = np.where(rand_img > 0)



        # Do it again: We'll call this random image the 'data'
        for l in range(0,n_sources):
            for m in range(0,len(img3)):
                if (var_2[l] > weight_inner[m]) and (var_2[l] < weight_outer[m]):
                    img3[m] = img3[m] + 1 # specifies flux of pixel.

        # Save random image to file:
        rand_img2 = np.reshape(img3,(n_dim,n_dim))
        here3 = np.where(rand_img2 > 0)


        # Find points in rand img 1
        rand_x = here2[1] # image position of x values
        rand_y = here2[0] # image position of y vales





        # Find points in rand img 2
        data_x = here3[1]
        data_y = here3[0] # image position of y vales



        plt.style.use('dark_background')
        plt.figure(figsize = [8, 8])
        plt.scatter(rand_x,rand_y, marker = '.', color = 'white', s = 25)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.title('Field '+field[target][0]+ ': Random Image')
        plt.savefig(path2 + '/rand_img.png')
        plt.close()
        # Save data image to file:
        plt.style.use('dark_background')
        plt.figure(figsize = [8, 8])
        plt.scatter(data_x,data_y, marker = '.', color = 'white', s = 25)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.title('Field '+field[target][0]+ ': Data Image')
        plt.savefig(path2 + '/data_img.png')
        plt.close()
        print("Random Image" + str(i+1) + " created...")

        # #############################################
        #Update: 2/9
        # Save source positions to file
        # Save data to file
        np.savetxt(path2+'/data.txt',np.transpose([data_x,data_y]),header = 'IMG_X IMG_Y',comments = '')
        np.savetxt(path2+'/rand.txt',np.transpose([rand_x,rand_y]),header = 'IMG_X IMG_Y',comments = '')




        # Begin calculating correlation function

        '''
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          Calculate Eudlidean distance between pairs
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        if (len(data_x) > 1) and (len(rand_x) > 1):
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





            # Bin
            dd_1 = np.histogram(dist_dd, bins = bins_lin)[0] /2
            dr_1 = np.histogram(dist_dr, bins = bins_lin)[0] /2
            rr_1 = np.histogram(dist_rr, bins = bins_lin)[0] /2



            # Compute corrleation fuction

            N_d = len(data_x)
            N_r = len(rand_x)
            N_corr = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))












            corr_lin[i] = W_ham(N_corr,dd_1,dr_1,rr_1)
            varr_lin[i] = 3*(1+(corr_lin[i])**2) / dd_1

            # Plot
            plt.style.use('default')
            plt.figure(figsize = [5,4], dpi = 300)
            centers_angle = centers_lin*pix_scale
            plt.errorbar(centers_angle, corr_lin[i], yerr=np.sqrt(varr_lin[i]), fmt='ko', elinewidth = 0.3, capsize = 3,ms = 4, mew =0.3, mfc = 'none')
            plt.xlabel('Separation (Arcsec)')
            plt.ylabel(r'W$(\theta)$')
            plt.title('Field '+field[target][0]+ ': 2paCF')
            plt.savefig(path2+'/corrfunc_raw.png')
            plt.close()



            # Tally
            rand_counts[i] = N_r
            data_counts[i] = N_d


            # Tally ratio
            ratio.append(len(data_x)/len(rand_x))




        # Flag ratio
        if (N_r / N_d) >= 2:
            flag[i] = 1

        elif (N_r / N_d)  <= 0.4:
            flag[i] = 2

        # Flag counts
        if (N_r < 2) or (N_d < 2):
            flag[i] = 3








 ################################################################
# Flag fields that have NaN's:
nan_check_corr_lin = np.isnan(corr_lin)
for i in range(0,len(corr_lin)):
    if 1 in nan_check_corr_lin[i]:
        flag[i] = 4



inf_check_corr_lin = np.isinf(corr_lin)
for i in range(0,len(corr_lin)):
    if 1 in inf_check_corr_lin[i]:
        flag[i] = 4



inf_check_varr_lin = np.isinf(varr_lin)
for i in range(0,len(varr_lin)):
    if 1 in inf_check_varr_lin[i]:
        flag[i] = 4




'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Flag legend:
    1: Too many random sources in field
    2: Too few random sources in field
    3: Less than 2 sources (data and random) in field
    4: Infs or NaNs in correlation function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''


# Select unflagged fields
check = np.where(flag == 0)[0]

corr_flag_lin = corr_lin[check]
varr_flag_lin = varr_lin[check]



field_count = len(check)

# Evaluate mean
corr_mean_lin = np.mean(corr_flag_lin, axis = 0)
varr_mean_lin = np.mean(varr_flag_lin, axis = 0)

# Write to file
np.savez(path1+'corr_raw',corr = corr_flag_lin, varr = varr_flag_lin)
np.savetxt(path1+'corr_mean.txt',np.transpose([centers_lin,corr_mean_lin,varr_mean_lin]),header = 'BIN_CENTERS(PIX) W VARR',comments = '')
np.savetxt(path1+ 'ratio.txt', (ratio), delimiter = ',')
np.savetxt(path1+'counts.txt',np.transpose([data_counts,rand_counts]),header = 'DATA_COUNTS RAND_COUNTS',comments = '')


print("correlation analysis complete! Have a great day. You're going to kill it.")
