'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script will:
    - Pipe in random and real data points for each exposure map
    - Calculate pair separations and bin them
    - Compute the Correlation Function
    - Evaluate the weighted average (and error) for each bin
    
For questions or comments, contact me at brett.s.bonine-1@ou.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

######################################################
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Begin Main Program
'''
#######################################################################
# Read in field list
path_fields = '/Users/bbonine/ou/research/corr_func/data/'
cat = path_fields + 'agntable_total.txt'
field = np.loadtxt(cat, dtype = str,delimiter = None, skiprows = 1, usecols=(15) , unpack = True)
data_xvals,data_yvals = np.loadtxt(cat, delimiter = None, skiprows = 1, usecols=(16,17) , unpack = True)

# Get rid of duplicates
field_list = np.unique(field)


# Read in random catalog
path_rand = path_fields + 'catalogs/'
grb = np.genfromtxt(path_rand+'random_cat.txt', skip_header = 1, usecols = (0), dtype = np.str)
rand_xvals,rand_yvals= np.genfromtxt(path_rand+'random_cat.txt', skip_header = 1,unpack = True, usecols = (1,2))

# Read in random catalog 2: this will be the data (delete for real data )
#grb_2 = np.genfromtxt(path_rand+'random_cat_2.txt', skip_header = 1, usecols = (0), dtype = np.str)
#data_xvals,data_yvals= np.genfromtxt(path_rand+'random_cat_2.txt', skip_header = 1,unpack = True, usecols = (1,2))


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
num_bins = int(input("Desired Number of bins: "))
bins = np.logspace(0.9,2.7,num_bins)
#bins = np.linspace(0,600,num_bins+1)
centers = 0.5*(bins[1:]+ bins[:-1])

# Initialize output arrays
Nrand_r = 0
corr = [] # corrfunc
varr = []
ratio_all = np.zeros(len(field_list))

corr_mean = np.zeros(len(bins)-1) # Final weighted correlation func
sig_mean = np.zeros(len(bins)-1) # Final weighted error
bin_fields = np.zeros(len(bins)-1) # Final tally of fields in bin

count_rand = np.zeros(len(field_list)) # array to check number of sources in each field
count_data = np.zeros(len(field_list))
field_count = 0

'''
Main Loop:

'''
###################################################################
cut = int(input("Cut on minimum number of sources to use: "))

path_main =  '/Users/bbonine/ou/research/corr_func/outputs/all/04_04_21/'
path_out = path_main+"nbins_"+str(num_bins)+"/"

if os.path.isdir(path_out) == False:
    os.mkdir(path_out)
    print("Output Directory created!")

f = open(path_out+ "ratio_all.txt", "a")
f.write("N_R\tN_D\tR/D\n")

loops = input("Desired Number of Fields to Analyze (" + str(len(field_list)) + " total): ")
while (int(loops) > len(field_list)) or (int(loops) < 1):
    loops = input("ERROR! Please input a valid number of iterations: " )

for i in range(0,int(loops)):

    # Select data for field 
    target = field_list[i]
    
    data_x,data_y = data_xvals[field==target], data_yvals[field==target]

    #For two random cats:
    #data_x, data_y = data_xvals[grb_2==target], data_yvals[grb_2==target]
    rand_x, rand_y = rand_xvals[grb == target],rand_yvals[grb==target]
    
    # Tally source counts
    count_data[i] = len(data_x)
    count_rand[i] = len(rand_x)
    # Check for empty values
    if (count_data[i] >= cut ) & (count_rand[i] >= cut):
        
        temp_1 = np.zeros(len(bins)-1)
        temp_2 = np.zeros(len(bins)-1)
        f.write(repr(len(rand_x))+"\t"+repr(len(data_x))+"\t"+repr(len(rand_x)/len(data_x))+"\n")
        ######################################################
        # Tally pair counts:
        print("Tallying pair counts for field " + target)
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
        dd_1 = np.histogram(dist_dd, bins = bins)[0] /2
        dr_1 = np.histogram(dist_dr, bins = bins)[0] /2
        rr_1 = np.histogram(dist_rr, bins = bins)[0] /2



        # Correlation Function Normalization:
        N_d = len(data_x)
        N_r = len(rand_x)
        N_corr = (N_d*N_r)**2 / ((N_d*(N_d-1)) * (N_r*(N_r-1)))

        # Evaluate Correlation Function

        # Evaluate Correlation Function
        temp1 = W_ham(N_corr,dd_1,dr_1,rr_1)
        
        # Variance 
        temp2 = 3*(1+(temp1)**2) / dd_1


        # Append results
        if len(corr) == 0:
            corr = temp1
            varr = temp2
        else:
            corr = np.vstack((corr,temp1))
            varr = np.vstack((varr,temp2))
        
        field_count += 1
f.close()
print(str(field_count) + " fields have minimum number of sources.")
print("Raw correlation functions calculated. Cleaning...")

'''
End Main Loop
'''
#############################################################
#Begin NaN and inf check for each bin
# Loop through each column
for i in range(0,len(varr[0])):
    flag = np.zeros(len(varr)) # null array to identify nan or infs
    inf_check = np.isinf(varr[:,i])
    nan_check = np.isnan(varr[:,i])


    # Identify values that fail the Check.
    # Also discard entries with a varience of zero.
    good_vals = np.where((inf_check != 1) & (nan_check != 1) & (varr[:,i] != 0))[0]

    # Select them
    corr_clean = corr[:,i][good_vals]
    varr_clean = varr[:,i][good_vals]

    # Evaluate weighted mean
    corr_mean[i],sig_mean[i] = weight(corr_clean,varr_clean)
    bin_fields[i] = len(corr_clean)
    print(np.mean(corr_clean))

print("AGN Analysis Complete! ")


print("Saving Outputs...")

np.savetxt(path_out+'out.txt',np.transpose([centers,corr_mean,sig_mean,bin_fields]),header = 'BIN_CENTERS(PIX) CORR SIG FIELDS',comments = '')


print("-------------------------------------------SUMMARY-------------------------------------------")
print(" ")
print("Minimum Number of Sources: " +str(cut))
print("Output files saved to " + path_main)
print("Correlation Analysis Complete! Have a nice day.")
print("---------------------------------------------------------------------------------------------")