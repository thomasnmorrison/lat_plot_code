# zeta_moments_plot.py

# Script to plot zeta moments
# Plot 1:
# Plot 2:
# Plot 3:
# Plot 4:

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = ''
en_bl_f = '' # Baseline run lattice averaged quantities
en_f = [''] # lattice averaged quatities
zm_bl_f = '' # Basline run zeta moments file
zm_f = [''] # zeta moments file

# Plot output file names


# Read in data
en_bl = np.loadtxt(path_n+en_bl_f, usecols = )
zm_bl = np.loadtxt(path_n+zm_bl_f, usecols =)

zm = np.zeros((np.shape(zm_f) + np.shape(zm_bl)))
en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
for i in range(0,len(zm_f)):
	en[i] = np.loadtxt(path_n+en_f[i], usecols =)
	zm[i] = np.loadtxt(path_n+zm_f[i], usecols =)


# Define indexing parameters


# Make plots
nfig = 0

# Plot 1:
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']

# Plot 2: 
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']

# Plot 3:
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']

# Plot 4:
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']


plt.show()
