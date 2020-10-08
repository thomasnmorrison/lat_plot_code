# fld_trajectory_plot.py

# Script to plot the field and field momentum trajectories
# Plot 1: \delta\phi clocked on \phi, \delta\phi clocked on \alpha 
# Plot 2: \delta\Pi_phi clocked on \phi, \delta\Pi_\phi clocked on \alpha 
# Plot 3: \delta\chi clocked on \phi, \delta\chi clocked on \alpha
# Plot 4: \delta\Pi_phi clocked on \phi, \delta\Pi_\phi clocked on \alpha

# to do:

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_TESTING_.out' # Baseline run lattice averaged quantities
en_f = [] # lattice averaged quatities
phi_bl_f = 'phi_lat_TESTING_.out'
phi_f = []
dphi_bl_f = 'dphi_lat_TESTING_.out'
dphi_f = []
chi_bl_f = 'chi_lat_TESTING_.out'
chi_f = []
dchi_bl_f = 'dchi_lat_TESTING_.out'
dchi_f = []

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
en_bl = np.loadtxt(path_n+en_bl_f)  # Formatted [time, column]
phi_bl = np.fromfile(path_n+phi_bl_f, dtype=np.double , count=-1)
dphi_bl = np.fromfile(path_n+dphi_bl_f, dtype=np.double , count=-1)
en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
phi = np.zeros((np.shape(phi_f) + np.shape(phi_bl)))
dphi = np.zeros((np.shape(dphi_f) + np.shape(dphi_bl)))

for i in range(0,len(en_f)):
	en[i] = np.fromfile(path_n+en_f[i], dtype=np.double , count=-1)
	phi[i] = np.fromfile(path_n+phi_f[i], dtype=np.double , count=-1)

if (chi_bl_f != ''):
	chi_bl = np.fromfile(path_n+chi_bl_f, dtype=np.double , count=-1)
	dchi_bl = np.fromfile(path_n+dchi_bl_f, dtype=np.double , count=-1)
	chi = np.zeros((np.shape(chi_f) + np.shape(chi_bl)))
	dchi = np.zeros((np.shape(dchi_f) + np.shape(chi_bl)))
	for i in range(0,len(en_f)):
		chi[i] = np.fromfile(path_n+chi_f[i], dtype=np.double , count=-1)
		dchi[i] = np.fromfile(path_n+dchi_f[i], dtype=np.double , count=-1)
 
# Reshape lattice data
print('np.shape(phi_bl) = ', np.shape(phi_bl))
print('np.shape(en_bl) = ', np.shape(en_bl))
nlat = nx*ny*nz
t =  int(len(en_bl[:,0]-1)/sl) + 1
print('t = ', t)
phi_bl = np.resize(phi_bl,(t,nlat)); phi_bl = phi_bl.transpose()       # Formatted [lat site, time]
dphi_bl = np.resize(dphi_bl,(t,nlat)); dphi_bl = dphi_bl.transpose()   # Formatted [lat site, time]
if (chi_bl_f != ''):
	chi_bl = np.resize(chi_bl,(t,nlat)); chi_bl = chi_bl.transpose()     # Formatted [lat site, time]
	dchi_bl = np.resize(dchi_bl,(t,nlat)); dchi_bl = dchi_bl.transpose() # Formatted [lat site, time]

print('np.shape(phi_bl) = ', np.shape(phi_bl))
#print('phi_bl[:,1] = ', phi_bl[:,1])


# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8
phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12

# Make plots
nfig = 0

# Plot 1: \delta\phi clocked on \phi, \delta\phi clocked on \alpha 
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\phi M_{Pl}^{-1}$', r'$\delta\phi M_{Pl}^{-1}$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

delta_f = phi_bl[:,:]-en_bl[::sl,phi_i]; delta_f = delta_f.transpose() # Formatted [time, lat site]
ax[0].plot(np.log(en_bl[::sl,a_i]), delta_f)
ax[1].plot(en_bl[::sl,phi_i], delta_f)
ax[1].invert_xaxis()


if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

# Plot 2: \delta\Pi_phi clocked on \phi, \delta\Pi_\phi clocked on \alpha 
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = [r'$\phi-\bar{\phi}$', r'$\phi-\bar{\phi}$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\dot{\phi} m_\phi^{-1}M_{Pl}^{-1}$', r'$\delta\dot{\phi} m_\phi^{-1}M_{Pl}^{-1}$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

delta_f = (dphi_bl[:,:]-en_bl[::sl,dphi_i])/en_bl[::sl,a_i]**3
delta_f = delta_f.transpose() # Formatted [time, lat site]
ax[0].plot(np.log(en_bl[::sl,a_i]), delta_f)
ax[1].plot(en_bl[::sl,phi_i], delta_f)
ax[1].invert_xaxis()

if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

if (chi_bl_f != ''):
# Plot 3: \delta\chi clocked on \phi, \delta\chi clocked on \alpha
	nfig += 1
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r'$\chi-\bar{\chi}$'
	s_title = [r'$\chi-\bar{\chi}$', r'$\chi-\bar{\chi}$']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
	y_lab = [r'$\delta\chi M_{Pl}^{-1}$', r'$\delta\chi M_{Pl}^{-1}$']
	fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	delta_f = chi_bl[:,:]-en_bl[::sl,chi_i]#; delta_f = delta_f.transpose() # Formatted [time, lat site]
	ax[0].plot(np.log(en_bl[::sl,a_i]), delta_f.T)
	ax[1].plot(en_bl[::sl,phi_i], delta_f.T)
	ax[1].invert_xaxis()

	if (SCALE_FIG == True):
		fig.set_size_inches(FIG_SIZE)
	if (SAVE_FIG[nfig-1] == True):
		fig.savefig()

if (dchi_bl_f != ''):
# Plot 4: \delta\Pi_phi clocked on \phi, \delta\Pi_\phi clocked on \alpha
	nfig += 1
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r'$\dot{\chi}-\dot{\bar{\chi}}$'
	s_title = [r'$\dot{\chi}-\dot{\bar{\chi}}$', r'$\dot{\chi}-\dot{\bar{\chi}}$']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
	y_lab = [r'$\delta\chi M_{Pl}^{-1}$', r'$\delta\chi M_{Pl}^{-1}$']
	fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	delta_f = (dchi_bl[:,:]-en_bl[::sl,dchi_i])/en_bl[::sl,a_i]**3        # Formatted [time, lat site]
	ax[0].plot(np.log(en_bl[::sl,a_i]), delta_f.T)
	ax[1].plot(en_bl[::sl,phi_i], delta_f.T)
	ax[1].invert_xaxis()

	if (SCALE_FIG == True):
		fig.set_size_inches(FIG_SIZE)
	if (SAVE_FIG[nfig-1] == True):
		fig.savefig()

plt.show()
