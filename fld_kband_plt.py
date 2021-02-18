# fld_kband_plt.py

# Script to make plots of the fields and momenta binned by k-band.

# Plot 1: Plot of 2d slice of fields binned by k-band at various time slices.

# to do: select time slice indicies
# to do: move kband function into a module
# to do: select k bands and put in k_ind

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from plot_param import *
from kband_mod import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read data
en = load_ener(en_f[0], path[0])
phi = load_lat(phi_f[0], path[0])
chi = load_lat(chi_f[0], path[0])
dphi = load_lat(dphi_f[0], path[0])
dchi = load_lat(dchi_f[0], path[0])

# Organize data
phi = np.reshape(phi, (nl,nx,ny,nz))
chi = np.reshape(chi, (nl,nx,ny,nz))
dphi = np.reshape(dphi, (nl,nx,ny,nz))
dchi = np.reshape(dchi, (nl,nx,ny,nz))
fld = np.array([[phi,dphi],[chi,dchi]])  # Formatted [fld,q/p,t,x,y,z]

# Select time slices and space slice
t_ind = [0,5,10]
k_ind = []
z_ind = 32

# Calculate FFT and band weights, banded fft and invert
fld_fft = np.fft.rfftn(fld[:,:,t_ind,:,:,:], axes=(-3,-2,-1))
W = band_w_lin(k_ind,(nx,ny,nz))
fld_fft_band = kband(fld_fft, W[:,:,:,:nz//2+1], (nx,ny,nz))  # Formatted [fld,q/p,t,band,i,j,k]
fld_band = np.fft.irfftn(fld_fft_band, axis=(-3,-2-,1))  # Formatted [fld,q/p,t,band,x,y,z]

# Make plots
nfig = 0

# Plot 1: Plot of 2d slices of \phi
nfig += 1




if SHOW_FIGS:
    plt.show()




