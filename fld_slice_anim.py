# fld_slice_anim.py

# Script to make an animation of field fluctuations on a 2d slice of the lattice.

# Animation 1: 

# to do: make animation
# to do: make an equivalent script that used VisIt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import potential as pot
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save/show figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read in data
en = load_ener(en_f[0], path[0])
phi = load_lat(phi_f[0], path[0])
chi = load_lat(chi_f[0], path[0])

# Calculate \phi fluctuations
phi = phi - en[:,phi_i]

# Animation

# Point of view
elev = 45.
azim = 45.
dist = 10.

# Save/show
if SAVE_FIGS:

if SHOW_FIGS:
    
