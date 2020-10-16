# swift_corrfunc
A reupload of my work on determining the two point angular correlation with the SWIFT AGN and Cluster Survey

'agn.py' is the primary script. This script iterates through fields in the agn catalog text file and then pulls the associated exposure map from those fields. 
Each exposure map is used to generate an associated synthetic SWIFt XRT image from that field. In the future, this will likely be its own script.
The synthetic images are then used to calculate the two-point correlation function based off the locations of sources also containe din the agn catalog.

'fit_practic.py' reads in the output of 'agn.py' and preforms a least-square minimization fit to the data. 

'deproj.py' is a work in progress, but will eventually run an MCMC algorithm in order to parameterize Limber's Inversion and extract the real-space clustering scale.
