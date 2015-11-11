
import itertools
import sys
import os
import numpy as np
from classy import Class
from copy import deepcopy
import matplotlib.pyplot as plt

# This simple demo tests the new features to store the N(z)s,
# and compares the Gaussian Mixture and histogram storage modes

class GaussianMixture():
	# Just a simple Gaussian mixture model
	def __init__(self, amps, means, stddevs):
		self.means = means
		self.amps = amps
		self.stddevs = stddevs
	# Return the number of gaussians
	def size(self):
		return len(self.amps) # assuming the lengths are consistent...
	# Evaluate on a grid
	def eval(self, grid):
		return np.sum([amp*np.exp(-0.5*((mu-grid)/sigma)**2.0)/np.sqrt(2*np.pi)/sigma 
			for amp, mu, sigma in zip(self.amps, self.means, self.stddevs)], axis=0)

# Construct 2 redshift samples with simple, overlapping N(z)s defined as Gaussian mixtures
redshiftdistributions = []
redshiftdistributions += [GaussianMixture([1.0], [0.9], [0.1])]
redshiftdistributions += [GaussianMixture([1.0, 0.1], [0.5, 0.6], [0.1, 0.1])]

# The main parameters for the Class run
mainparams = dict()
mainparams.update({ 
    'output': 'tCl,pCl,lCl,nCl,sCl,dCl',
    'lensing': 'yes',
    'l_max_scalars': 100,
    'l_max_lss': 100,
    'non_diagonal': 1,
    'selection_num': len(redshiftdistributions), # Two samples
    'bias_1': 1.0, # Bias of first
    'bias_2': 1.5, # Bias of second
    's_bias_1': 0.4, # Magnification bias of first
    's_bias_2': 0.4, # Magnification bias of second
    })

# Scenario 1 : storing the N(z)s as Gaussian mixtures
scenario1 = dict()
scenario1.update({
	'selection': 'multigaussian' # The N(z)s are described with Gaussian mixtures
	})
for i, nz in enumerate(redshiftdistributions):
	ic = str(i+1)
	scenario1.update({ 
	    'selection_'+ic+'_num': nz.size(), # Number of gaussians for the N(z) of the i-th sample
	})
	for j in range(nz.size()):
		jc = str(j+1)
		scenario1.update({
		    'selection_'+ic+'_amp_'+jc: nz.amps[j], # Amplitude of the j-th gaussian of the i-th sample
		    'selection_'+ic+'_mean_'+jc: nz.means[j], # Mean of the j-th gaussian of the i-th sample
		    'selection_'+ic+'_width_'+jc: nz.stddevs[j] # Stddev of the j-th gaussian of the i-th sample
	    })

# Scenario 2 : binning the N(z)s into histograms and passing txt files to Class
scenario2 = dict()
scenario2.update({
	'selection': 'histogram' # The N(z)s are described with histograms in text files
})
z_grid = np.linspace(0, 2, num=100)
for i, nz in enumerate(redshiftdistributions):
	ic = str(i+1)
	fname = 'test_class2_nz_'+ic+'.txt'
	nz_grid = nz.eval(z_grid) # Evaluate N(z) on grid
	np.savetxt(fname, np.vstack((z_grid, nz_grid)).T) # Store N(z) to file
	scenario2.update({ 
	    'selection_'+ic+'_file': fname, # File for the N(z) histogram of the i-th sample
	})

if False: # In case you want to plot the N(z)s
	fig, axs = plt.subplots(1, 2, figsize=(14, 5))
	finer_z_grid = np.linspace(0, 2, num=2000)
	for i, nz in enumerate(redshiftdistributions):
		ic = str(i+1)
		nz_grid = nz.eval(z_grid)
		finer_nz_grid = nz.eval(finer_z_grid)
		axs[i].plot(finer_z_grid, finer_nz_grid, label='Gaussian Mixture')
		axs[i].plot(z_grid, nz_grid, label='Histogram-ized', ls='steps')
	plt.show()

# Now run Class!
cosmo = Class()
# Scenario 1
cosmo.set(dict(mainparams.items()+scenario1.items()))
cosmo.compute()
cl1 = cosmo.density_cl(mainparams['l_max_lss'])
cosmo.struct_cleanup()
cosmo.empty()
# Scenario 2
cosmo.set(dict(mainparams.items()+scenario2.items()))
cosmo.compute()
cl2 = cosmo.density_cl(mainparams['l_max_lss'])
cosmo.struct_cleanup()
cosmo.empty()

# The Cls should be very close if the histogram is binned finely
nbins = len(redshiftdistributions)
print 'Comparing accuracy of N(z) representation: multigaussian vs histograms'
for i in range(nbins*(nbins+1)/2): 
	err = cl1['dd'][i] / cl2['dd'][i]
	print 'Accuracy of density Cls ', i+1
	print 'Mean and stddev on ratio of Cls:', np.mean(err[2:]), np.std(err[2:])
