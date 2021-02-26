'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script will:
    - Pipe in random and real data points for each exposure map
    - Calculate pair separations and bin them
    - Compute the Correlation Function
    - Evaluate the weighted average (and error) for each bin
    - Plot and save outputs

For questions or comments, contact me at brett.s.bonine-1@ou.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

######################################################
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import os

####################################################
# Matplotlib stuff
fontsize = 8
figsize = (3,3)
dpi = 300

# Configure parameters
plt.rcParams.update({'font.size': fontsize, 'figure.figsize': figsize, 'figure.dpi': dpi})

# Default tick label size
plt.rcParams['text.usetex'] = False
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1


'''
Begin Main Program
'''
#######################################################################
# Read in field list
path_fields = '/Users/bbonine/ou/research/corr_func/data/'
cat = path_fields + 'agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
x,y = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(16,17) , unpack = True)

# Get rid of duplicates
field_list = np.unique(field)


###################################################
# Read in pair tallys
path_main = '/Users/bbonine/ou/research/corr_func/outputs/02_15_21_rand_15bin/'
data_counts,rand_counts = np.loadtxt(path_main +'counts.txt', unpack = True, skiprows = 1)

###################################################
# Find out how many field meet the count cut
cut = 5
count_check = np.where((data_counts >= cut) & (rand_counts >= cut))[0]

field_cut = len(count_check)
print(field_cut)


######################################################
#Functions
def distance(x2,x1,y2,y1):
    return (((x2-x1)**2 + (y2-y1)**2)**0.5) # Euclidiean distance

def W_ham(N,DD,DR,RR):
    return (N*(( DD* RR) / (DR)**2) ) -1 # Hamilton Correlation Estimator


def weight(corr,varr):                   # Weighted Average
    # Evaluate Weighted mean
    corr_mu = np.sum(corr/varr) / np.sum(1/varr)
    # Evaluate Standard error of the weighted mean
    sig_mu = np.sqrt(1/(np.sum(varr**(-1))))

    return corr_mu, sig_mu

# Binning
num_bins = 20
bins_lin = np.logspace(1,2.9,num_bins)
centers_lin = 0.5*(bins_lin[1:]+ bins_lin[:-1])

# Null arrays
# Size determined by how many fields meet the cut
loops = field_cut
corr_lin = np.zeros((loops,len(bins_lin)-1)) # corrfunc
varr_lin = np.zeros((loops,len(bins_lin)-1)) # varrience

corr_mean = np.zeros(len(bins_lin)-1) # Final weighted correlation func
sig_mean = np.zeros(len(bins_lin)-1) # Final weighted error
bin_fields = np.zeros(len(bins_lin)-1) # Final tally of fields in bin

'''
Main Loop:

'''
print("Talling pair counts...")
for i in range(0,loops):
    # Check for empty values
    if (data_counts[i] and rand_counts[i]) > 5:


        path_target = path_main+field_list[i]

        # Read in values
        # For random 'data':
        #data_x, data_y = np.loadtxt(path_target+'/data.txt',unpack = True, skiprows =1)
        # For real data:
        data = np.where(field == field_list[i])[0]
        data_x,data_y = x[data], y[data]
        rand_x, rand_y = np.loadtxt(path_target+'/rand.txt',unpack = True, skiprows = 1)



        ######################################################
        # Tally pair counts:

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



        # Correlation Function Normalization:
        N_d = len(data_x)
        N_r = len(rand_x)
        N_corr = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))

        # Evaluate Correlation Function

        corr_lin[i] = W_ham(N_corr,dd_1,dr_1,rr_1)
        varr_lin[i] = 3*(1+(corr_lin[i])**2) / dd_1

print("Raw correlation functions calculated. Cleaning...")

'''
End Main Loop
'''

#############################################################
#Begin NaN and inf check for each bin
# Loop through each column
for i in range(0,len(varr_lin[0])):
    flag = np.zeros(len(varr_lin)) # null array to identify nan or infs
    inf_check = np.isinf(varr_lin[:,i])
    nan_check = np.isnan(varr_lin[:,i])


    # Identify values that fail the Check.
    # Also discard entries with a varience of zero.
    good_vals = np.where((inf_check != 1) & (nan_check != 1) & (varr_lin[:,i] != 0))[0]

    # Select them
    corr = corr_lin[:,i][good_vals]
    varr = varr_lin[:,i][good_vals]

    # Evaluate weighted mean
    corr_mean[i],sig_mean[i] = weight(corr,varr)
    bin_fields[i] = len(corr)

print("Saving Outputs...")

path_output = '/Users/bbonine/ou/research/corr_func/figures/02_20_21_data_log/'
# Make output folder
if os.path.isdir(path_output) == False:
    os.mkdir(path_output)


#Convert pixel separation to angular separation
pix_scale = 47.1262 / 20      # arcseconds / pixel for SWIFT XRT
centers_angle = centers_lin * pix_scale

# Plot
plt.figure(figsize = [6,4], dpi = 300)
plt.errorbar(centers_angle, corr_mean, yerr=sig_mean, fmt='ko', elinewidth = 0.4, capsize = 1,ms = 2, mew =0.3)
plt.xlabel('Separation (Arcsec)')
plt.ylabel(r'W$(\theta)$')
plt.title('2paCF: Two Random Images')
plt.text(200,-0.04, r'$\mu = $' +str(np.around(np.mean(corr_mean),3)))
plt.yscale('log')
plt.xscale('log')
plt.savefig(path_output+'corr_weight_mean.png')

# Save info to text files
# Save random image to file
np.savetxt(path_output+'out.txt',np.transpose([centers_lin,corr_mean,sig_mean,bin_fields]),header = 'BIN_CENTERS(PIX) CORR SIG FIELDS',comments = '')

print("Correlation Analysis Complete!")
