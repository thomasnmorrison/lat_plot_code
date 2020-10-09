# fld_real_space_plot.py

# Script to plot real space slices of the fields
# Plot 1: Colormap of \phi on a 2d slice at specified time slices
# Plot 2: Colormap of \dot{\phi} on a 2d slice at specified time slices
# Plot 3: 

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
dx = 10./nx

# Configuration parameters
SAVE_FIG = [False, False, False]
SCALE_FIG = [False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False]

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14
anno_fs = 12

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
t =  int(len(en_bl[:,0]-1)/sl) + 1
phi_bl = np.resize(phi_bl,(t,nx,ny,nz))#; phi_bl = phi_bl.transpose()       # Formatted [time, x, y, z]
dphi_bl = np.resize(dphi_bl,(t,nx,ny,nz))#; dphi_bl = dphi_bl.transpose()   # Formatted [time, x, y, z]
if (chi_bl_f != ''):
	chi_bl = np.resize(chi_bl,(t,nx,ny,nz))#; chi_bl = chi_bl.transpose()     # Formatted [time, x, y, z]
	dchi_bl = np.resize(dchi_bl,(t,nx,ny,nz))#; dchi_bl = dchi_bl.transpose() # Formatted [time, x, y, z]

# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8
phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12

# Calc H, \dot{\phi}
en_bl[:,hub_i] = np.sqrt(-en_bl[:,hub_i]/3.)

# Time slices
t_slice = [0,0,0,0]  # indicies to show slices

# Make plots
nfig = 0

# Plot 1: Colormap of \phi on a 2d slice at specified time slices
nfig += 1
nr = 1; nc = 4  # number of subplot rows and columns
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=False)
f_title = r'$\delta\phi$ Realspace Slice'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\phi M_{Pl}^{-1}$', r'$\delta\phi M_{Pl}^{-1}$']
fig.suptitle(f_title, fontsize = title_fs)

for i in range(0,nc):
	f_slice = ax[i].imshow(phi_bl[0,:,:,7]-en_bl[0*sl,phi_i], cmap='viridis', aspect='equal', origin='lower')
	ax[i].plot([0,1./(dx*en_bl[0,a_i]*en_bl[0,hub_i])],[0,0], c='w')
	ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w', fontsize=anno_fs)
	ax[i].set_yticks([]); ax[i].set_xticks([])
cax = fig.add_axes([])
fig.colorbar(f_slice, cax=cax)

#fig.set_tight_layout(True)
plt.show()
