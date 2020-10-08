# infl_param_plot.py

# Script to plot the inflation parameters
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


# Plot output file names


# Read in data
en_bl = np.loadtxt(path_n+en_bl_f, usecols = )

en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
for i in range(0,len(en_f)):
	en[i] = np.loadtxt(path_n+en_f[i], usecols =)

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

# Plot 5:
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']






plt.show()
