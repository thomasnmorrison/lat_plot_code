# zeta_phase_space_plot.py

# Script to plot the 'phase-space' of \zeta and \dot{\zeta}, and likewise for
# \zeta_{part} and \dot{\zeta}_{part}
# Plot 1: Trajectories through the \zeta \dot{\zeta} space
# Plot 2: 2d histograms of time slices of \zeta \dot{\zeta} space
# Plot 3: Contour plots of time slices of \zeta \dot{\zeta} space

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_TESTING_.out' # Baseline run lattice averaged quantities
en_f = [] # lattice averaged quatities
zeta_bl_f = ''
zeta_f = []

# Run parameters
nx = 8; ny = 8; nz = 8
sl = 2**2 # steplat
ds = 2**3 # down sampling of lattice to plot

# Configuration parameters
SAVE_FIG = [False, False, False]
SCALE_FIG = [False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False]

# Plot output file names
