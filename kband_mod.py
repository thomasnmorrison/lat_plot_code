# kband_mod.py

# Module containing function useful for putting lattice quantities into kbands.

# Import packages
import numpy as np

# Define functions

# Function to take a field and return it split into bands with specified k weighting.
# f: input field
# W: array of weights per k band
# n: lattice size tuple
def fldband(f,W,n):
    fk = np.fft.rfftn(f,n)
    print(np.shape(fk))
    fk_band = kband(fk, W, n)
    print(np.shape(fk_band))
    f_band = np.fft.irfftn(fk_band,n)
    print(np.shape(f_band))
    return f_band

# Function to take an fft and return it split into k-bands. Modes between bands are given
# linear weighting to neighbouring bands.
# fft_in: input fft
# W_in: input weights for each fft mode in each band
# n: shape of lattice as a tuple
def kband(fft_in, W_in, n):
    fft_kband = np.resize(fft_in, (len(W_in),) + np.shape(fft_in))  # new array that is a resize of fft_in with an additional axis for kband
    fft_kband = np.moveaxis(fft_kband, 0, -1-len(n))
    fft_kband = fft_kband * W_in
    return fft_kband

# Function to compute and return |k| on the lattice
# n: tuple of dimensions of lattice (not with the half length axis for the fft)
def lat_freq(n):
    ind_temp = np.indices(n)
    ind = np.zeros(np.shape(ind_temp))
    for i in range(0,len(ind)):
        ind[i] = np.where((ind_temp[i]<n[i]//2+1),ind_temp[i],ind_temp[i]-n[i])
    ind = ind**2
    rad = np.sqrt(np.sum(ind,axis=0))
    return rad

# Function to compute weights for k-bands with a linear weighting fo modes between bands.
# n.b. If this is to be used for a real fft you will need to manualy choose the short axis.
#      The default when using np.fft.rfftn is for the last axis to be short.
# k_ind: k bands in units of \Delta k
# n: shape of lattice as a tuple
def band_w_lin(k_ind, n):
    W = np.zeros((len(k_ind),)+n)
    rad = lat_freq(n)  # radius in k space of each lattice site
    for i in range(0,len(k_ind)-1):
        W[i] += np.where((rad>=k_ind[i] and rad<k_ind[i+1]),
                         1.-(rad-k_ind[i])/(k_ind[i+1]-k_ind[i]),0.)
        W[i+1] += np.where((rad>=k_ind[i] and rad<k_ind[i+1]),
                           (rad-k_ind[i])/(k_ind[i+1]-k_ind[i]),0.)
    W[-1] += np.where((rad==k_ind[-1]),1.,0.)
    return W

# Function to generate weighting for a low pass gaussian filter.
# sig: sigma of gaussian in units of \Delta l
# n: shape of fft lattice as a tuple
def low_pass_gauss(sig,n):
    rad = lat_freq(n)
    rad = rad[...,:n[-1]//2+1]
    W = np.exp(-rad**2/(2*sig**2))
    return W
