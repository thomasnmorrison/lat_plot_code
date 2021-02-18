# dzeta_fld_space_plt.py

# Script for plots showing where in field space \dot{\zeta} is being produced.

# Plot 1:

# to do: calculate \Delta\frac{d\zeta}{d\alpha}=\Delta\dot{\zeta}/H
# to do: declare histogram limits
# to do: 

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg
import matplotlib.colors as colors
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read-in data
en = load_ener(ener_f[0], path[0])
en_bl = load_ener(ener_f_bl, path_bl)
dzeta = load_lat(zeta_f[0], path[0])
dzeta_bl = load_lat(zeta_f_bl, path_bl)
phi = load_lat(phi_f[0], path[0])
chi = load_lat(chi_f[0], path[0])
print('Data read-in complete')

# Create mesh to use with pcolormesh


# Define bins
nbins=2**8
chi_range = np.amax([np.absolute(np.amin(chi)),np.absolute(np.amax(chi))])
chi_bin = np.linspace(-chi_range, chi_range, num=nbins+1, endpoint=True)

# Process data
Delta_dzeta = dzeta - dzeta_bl  # Formatted [t, xyz]

# Sort data at each time slice according to \chi
chi_argsort = np.argsort(chi,axis=-1)
chi_sort = np.take_along_axis(chi, chi_argsort, 0)
phi_sort = np.take_along_axis(phi, chi_argsort, 0)
Delta_dzeta_sort = np.take_along_axis(Delta_dzeta, chi_argsort, 0)

#ind = np.zeros((nl,nbins+1))  # index at each time slice where array is partitioned when binning chi_sort
#Delta_dzeta_binned = np.zeros((nl,nbins))
#for i in range(0,nl):
#	ind[i] = np.searchsorted(chi_sort, chi_bin)

#index = (,)
#for i in range(0,nl):
#	temp = (,)
#	ind = np.searchsorted(chi_sort[i], chi_bin)
#	for j in range(0,nbins):
#		temp = temp + (np.arange(ind[j],ind[j+1]),)
#		#temp = temp + (np.arange(ind[i,j],ind[i,j+1]),)
#	temp = np.array(temp)
#	index = index + (temp,)
#index = np.array(index)  # [t,[bin,[ind in bin]]]

# to do: reshape index to be interated over 1d
# to do: create an array of \Delta\dot{\zeta} formatted as [t*nbins+bin,[binned data]] by using take_along_slice
# to do: find the mean \Delta\dot{\zeta} in each bin formatted as [t*nbins+bin,mean]
# to do: reformat as [t,bin,mean]

index = (,)
for i in range(0,nl):
	ind = np.searchsorted(chi_sort[i], chi_bin)
	for j in range(0,nbins):
		index = index + (np.arange(ind[j],ind[j+1]),)
index = np.array(index)  # [t*nbins+bin, [ind in bin at t]]




