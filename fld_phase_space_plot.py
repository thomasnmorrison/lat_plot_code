# fld_phase_space_plot.py

# Script to plot the phase space of fields and momenta
# Plot 1: Trajectories through the \phi \Pi_\phi space
# Plot 2: 2d histograms of time slices of \phi \Pi_\phi space
# Plot 3: Contour plots of time slices of \phi \Pi_\phi space

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
SAVE_FIG = [False, False, False]
SCALE_FIG = [False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False]

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14

# Plot output file names

# Read in data
en_bl = np.loadtxt(path_n+en_bl_f)
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

# Plot 1: Trajectories through the \phi \Pi_\phi space
nfig += 1
t = 0
nbins = 25
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'$\delta\phi\text{ vs }\delta\Pi_\phi$'
s_title = ['','']
x_lab = [r'']
y_lab = [r'']

f1 = phi_bl[:,:]-en_bl[::sl,phi_i]
f2 = dphi_bl[:,:]-en_bl[::sl,dphi_i]
ax.plot(f1[:,:].T, f2[:,:].T)


# Plot 2: 2d histogram of \phi and \Pi_\phi
# to do: Make 2d histogram of \phi \Pi_\phi slice of phase space
# to do: Sub plots of \phi-\Pi_\phi, \chi-\Pi_\chi, \phi-\chi, \Pi_\phi-\Pi_\chi, 
# to do: Make contour plot of of 2d histogram
# to do: Make this into an animation over time slices

nfig += 1
t = 0
nbins = 25
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\phi M_{Pl}^{-1}$', r'$\delta\phi M_{Pl}^{-1}$']
f1 = phi_bl[:,:]-en_bl[::sl,phi_i]
f2 = dphi_bl[:,:]-en_bl[::sl,dphi_i]
ax.hist2d(f1[:,t], f2[:,t], bins=nbins)

# Plot 3: contour plot of \phi and \Phi_\phi
# to do: this plot relies on Plot 1 as written, I can use np.hist2d to fix that
# to do: fix centering
nfig += 1
t = 0
nbins = 15
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\phi M_{Pl}^{-1}$', r'$\delta\phi M_{Pl}^{-1}$']
h, xbin, ybin = np.histogram2d(f1[:,t], f2[:,t], bins=nbins)
xbin_c = (xbin[:-1]+xbin[1:])/2.; ybin_c = (ybin[:-1]+ybin[1:])/2.
ax.contour(xbin_c, ybin_c, h, levels=np.arange(0,50,3))

if (chi_bl_f != ''):
# Plot 4: 2d histogram of \chi and \Pi_\chi
	nfig += 1
	t = 0
	nbins = 25
	fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
	f_title = r'$\chi, \Pi_\chi$ Density'
	s_title = ['','']
	x_lab = [r'$\chi M_{Pl}^{-1}$']
	y_lab = [r'$\Pi_\chi (m_\phi M_{Pl})^{-1}$']
	fig.suptitle(f_title, fontsize = title_fs)
	ax.set_xlabel(x_lab[0], fontsize = x_lab_fs)
	ax.set_ylabel(y_lab[0], fontsize = y_lab_fs)
	ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	f1 = chi_bl[:,:]#-en_bl[::sl,chi_i]
	f2 = dchi_bl[:,:]#-en_bl[::sl,dchi_i]
	ax.hist2d(f1[:,t], f2[:,t], bins=nbins)


# Plot 4:
plt.show()
