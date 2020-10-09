# zeta_trajectory_plot.py

# Script to plot \zeta and \zeta_{part} trajectories
# Plot 1: ln(a) vs \zeta
# Plot 2:

# to do: \zeta_{part} plots

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_TESTING_.out' # Baseline run lattice averaged quantities
en_f = ['energy_spec_TESTING_.out'] # lattice averaged quatities
zeta_bl_f = 'zeta_lat_TESTING_.out'
zeta_f = ['zeta_lat_TESTING_.out']

# Run parameters
nx = 8; ny = 8; nz = 8
sl = 2**2 # steplat
ds = 2**3 # down sampling of lattice to plot

# Configuration parameters
SAVE_FIG = [False, False, False, False]
SCALE_FIG = [False, False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False, False]

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14

# Plot output file names

# Read in data
en_bl = np.loadtxt(path_n+en_bl_f)
zeta_bl = np.fromfile(path_n+zeta_bl_f, dtype=np.double , count=-1)
en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
zeta = np.zeros((np.shape(zeta_f) + np.shape(zeta_bl)))

for i in range(0,len(en_f)):
	en[i] = np.loadtxt(path_n+en_f[i])
	zeta[i] = np.fromfile(path_n+zeta_f[i], dtype=np.double , count=-1)

# Reshape lattice data
nlat = nx*ny*nz
t =  int(len(en_bl[:,0]-1)/sl) + 1
zeta_bl = np.resize(zeta_bl,(t,nlat)); zeta_bl = zeta_bl.transpose()       # Formatted [lat site, time]
if (len(zeta_f) > 0):
	zeta = np.resize(zeta, (len(zeta_f),t,nlat)); zeta = zeta.transpose((0,2,1))  # Formatted [run, lat site, time]

# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8
phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12

# Make plots
nfig = 0

# Plot 1: \zeta clocked on ln(a)
nfig += 1
fig, ax = plt.subplots(nrows=2, ncols=1 , sharex=False)
f_title = r''
s_title = [r'$\zeta_{V_0}$', r'$\zeta_{\Delta V}$']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[0], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[0], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	ax[i].set_xmargin(0.)

ax[0].plot(np.log(en_bl[::sl,a_i]), zeta_bl.T)
for i in range(0,len(zeta_f)):
	ax[1].plot(np.log(en[i,::sl,a_i]), (zeta[i,:,:]).T)

fig.set_tight_layout(True)
if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

if (len(zeta_f) > 0):
# Plot 2: \Delta\zeta clocked on ln(a)
	nfig += 1
	fig, ax = plt.subplots(nrows=1, ncols=1 , sharex=False)
	f_title = r''
	s_title = [r'$\Delta\zeta$']
	x_lab = [r'$\mathrm{ln}(a)$']
	y_lab = [r'']
	#fig.suptitle(f_title, fontsize = title_fs)
	ax.set_title(s_title[0], fontsize = stitle_fs)
	ax.set_xlabel(x_lab[0], fontsize = x_lab_fs)
	ax.set_ylabel(y_lab[0], fontsize = y_lab_fs)
	ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	ax.set_xmargin(0.)

	for i in range(0,len(zeta_f)):
		ax.plot(np.log(en_bl[::sl,a_i]), (zeta[i,:,:]-zeta_bl[:,:]).T)

	fig.set_tight_layout(True)
	if (SCALE_FIG == True):
		fig.set_size_inches(FIG_SIZE)
	if (SAVE_FIG[nfig-1] == True):
		fig.savefig()







plt.show()
