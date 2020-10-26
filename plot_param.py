# plot_param.py

# Parameters for plot data. Eg. file names, paths, etc.
# Parameters for runs. Eg. nlat, sl, etc.

# to do: reshape field data before returning to [run, t, xyz] (not [run, t, x, y, z])

# Import packages
import numpy as np

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
run_indent_bl = '_TESTING32_.out'
run_indent = '_TESTING32_DV_.out'
en_bl_f = 'energy_spec_TESTING32_.out'   # Baseline run lattice averaged quantities
en_f = ['energy_spec_TESTING32_DV_.out'] # lattice averaged quatities
phi_bl_f = 'phi_lat_TESTING32_.out'      
phi_f = ['phi_lat_TESTING32_DV_.out']
dphi_bl_f = 'dphi_lat_TESTING32_.out'
dphi_f = ['dphi_lat_TESTING32_DV_.out']
chi_bl_f = 'chi_lat_TESTING32_.out'
chi_f = ['chi_lat_TESTING32_DV_.out']
dchi_bl_f = 'dchi_lat_TESTING32_.out'
dchi_f = ['dchi_lat_TESTING32_DV_.out']
zeta_bl_f = 'zeta_lat_TESTING32_.out'
zeta_f = ['zeta_lat_TESTING32_DV_.out']

# Lattice parameters
nx = 32; ny = 32; nz = 32
nlat = nx*ny*nz
llen = 1.75
dx = llen/nx
dk = 2*np.pi/llen
nfld = 2

# Data outputting parameters
sl = 2**2  # steplat
ss = 2**2  # stepspec
ds = 2**3  # down sampling of lattice to plot
ne = 257   # number of energy output steps
nl = (ne-1)//sl+1  # number of lattice output steps

# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8; phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12

# Colours
c_ar = ['b','r','g','k','y','c']

# File reading functions

# Function to read energies and mean fields from file
# en_bl Formatted [time, column]
# en Formatted [run, time, column]
def load_energy():
	# Load baseline energies
	en_bl = np.loadtxt(path_n+en_bl_f)
	# Load enegries from additional runs
	en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
	for i in range(0,len(en_f)):
		en[i] = np.loadtxt(path_n+en_f[i])
	return en_bl, en

# Function to read fields and momenta from file and resize the arrays
def load_fld():
	# Load baseline fields
	phi_bl = np.fromfile(path_n+phi_bl_f, dtype=np.double , count=-1)
	dphi_bl = np.fromfile(path_n+dphi_bl_f, dtype=np.double , count=-1)
	if (chi_bl_f != ''):
		chi_bl = np.fromfile(path_n+chi_bl_f, dtype=np.double , count=-1)
		dchi_bl = np.fromfile(path_n+dchi_bl_f, dtype=np.double , count=-1)
	# Load fields from additional runs
	phi = np.zeros((np.shape(phi_f) + np.shape(phi_bl)))
	dphi = np.zeros((np.shape(dphi_f) + np.shape(dphi_bl)))
	for i in range(0,len(phi_f)):
		phi[i] = np.fromfile(path_n+phi_f[i], dtype=np.double , count=-1)
		dphi[i] = np.fromfile(path_n+dphi_f[i], dtype=np.double , count=-1)
	if (chi_bl_f != ''):
		chi = np.zeros((np.shape(chi_f) + np.shape(chi_bl)))
		dchi = np.zeros((np.shape(dchi_f) + np.shape(chi_bl)))
		for i in range(0,len(chi_f)):
			chi[i] = np.fromfile(path_n+chi_f[i], dtype=np.double , count=-1)
			dchi[i] = np.fromfile(path_n+dchi_f[i], dtype=np.double , count=-1)
	# Resize data
	phi_bl = np.resize(phi_bl,(nl,nlat))      # Formatted [time, lat site]
	dphi_bl = np.resize(dphi_bl,(nl,nlat))    # Formatted [time, lat site]
	if (chi_bl_f != ''):
		chi_bl = np.resize(chi_bl,(nl,nlat))    # Formatted [time, lat site]
		dchi_bl = np.resize(dchi_bl,(nl,nlat))  # Formatted [time, lat site]
	if (len(phi_f) > 0):
		phi = np.resize(phi, (len(phi_f),nl,nlat))     # Formatted [run, time, lat site]
		dphi = np.resize(dphi, (len(dphi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (len(chi_f) > 0):
		chi = np.resize(chi, (len(chi_f),nl,nlat))     # Formatted [run, time, lat site]
		dchi = np.resize(dchi, (len(dchi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (chi_bl_f == ''):
		return phi_bl, dphi_bl, phi, dphi, 0, 0, 0, 0
	else:
		return phi_bl, dphi_bl, chi_bl, dchi_bl, phi, dphi, chi, dchi

# Function to read zeta from file and resize the arrays
def load_zeta():
	zeta_bl = np.fromfile(path_n+zeta_bl_f, dtype=np.double , count=-1)
	zeta = np.zeros((np.shape(zeta_f) + np.shape(zeta_bl)))
	for i in range(0,len(zeta_f)):
		zeta[i] = np.fromfile(path_n+zeta_f[i], dtype=np.double , count=-1)
	# resize data
	zeta_bl = np.resize(zeta_bl,(nl,nlat))      # Formatted [time, lat site]
	if (len(zeta_f) > 0):
		zeta = np.resize(zeta, (len(zeta_f),nl,nlat))     # Formatted [run, time, lat site]
	return zeta_bl, zeta






