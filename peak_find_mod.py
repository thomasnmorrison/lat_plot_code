# peak_finder_mod.py

# Module for finding peaks

import numpy as np

# Function to find peaks
# inputs
# f: fields in which to find peaks
# returns
# peak_p: array of shape f with entries True at maxima
# peak_n: array of shape f with entries True at minima
# peak_p_ind: indices of maxima
# peak_n_ind: indicies of minima
def peak_find_3d(f):
    peak_p = np.fill(np.shape(f), True)
    peak_n = np.fill(np.shape(f), True)
    neighbours = [-1,0,1]
    for i in neighbours:
        for j in neighbours:
            for k in neighbours:
                if (not(i==0 and j==0 and k==0)):
                    temp = np.roll(f,(i,j,k),axis=(-3,-2,-1))
                    peak_p = np.where((f > temp), peak_p, False)
                    peak_n = np.where((f < temp), peak_p, False)
    peak_p_ind = np.nonzero(peak_p)
    peak_n_ind = np.nonzero(peak_n)
    return peak_p, peak_n, peak_p_ind, peak_n_ind

# Function to return an ordering of peaks
# inputs
# returns
# val_p: ordered array of the values of maxima in f
# val_n: ordered array of the values of minima in f
def peak_height_3d(f):
    return val_p, val_n

# Find peaks                                                                                                          
print('Finding peaks')
fld = zeta[0,:,:,:,:] - zeta_bl[:,:,:,:]           # Formatted [t,x,y,z]                                              
peak_p_ar = np.ones(np.shape(fld), dtype=np.int8)  # Formatted [t,x,y,z]                                              
peak_n_ar = np.ones(np.shape(fld), dtype=np.int8)  # Formatted [t,x,y,z]                                              
for i in [-1,0,1]:
        for j in [-1,0,1]:
                for k in [-1,0,1]:
                        if (not(i==0 and j==0 and k==0)):
                                temp = np.roll(fld,(i,j,k),axis=(1,2,3))
                                peak_p_ar = np.where((fld > temp), peak_p_ar, 0)
                                peak_n_ar = np.where((fld < temp), peak_n_ar, 0)

# Make tupple of arrays of peak heights at each time                                                                  
peak_p_tuple = ()
peak_n_tuple = ()
for i in np.arange(0,len(fld[:])):
        peak_p_sort = fld[i,peak_p_ar[i].nonzero()[0],peak_p_ar[i].nonzero()[1],peak_p_ar[i].nonzero()[2]]
        peak_n_sort = fld[i,peak_n_ar[i].nonzero()[0],peak_n_ar[i].nonzero()[1],peak_n_ar[i].nonzero()[2]]
        peak_p_sort.flatten()
        peak_n_sort.flatten()
        peak_p_sort.sort()
        peak_n_sort.sort()
        peak_p_tuple = peak_p_tuple+(peak_p_sort,)  # Formatted [t][ordered height]                                   
        peak_n_tuple = peak_n_tuple+(peak_n_sort,)  # Formatted [t][ordered height]                                   

peak_p_tuple = np.array(peak_p_tuple)  # Formatted [t][ordered height]                                                
peak_n_tuple = np.array(peak_n_tuple)  # Formatted [t][ordered height]                                                
print('np.shape(peak_p_tuple[-1]) = ', np.shape(peak_p_tuple[-1]
