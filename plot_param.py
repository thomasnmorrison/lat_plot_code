# plot_param.py

# Parameters for plot data. Eg. file names, paths, etc.
# Parameters for runs. Eg. nlat, sl, etc.

# to do: reshape field data before returning to [run, t, xyz] (not [run, t, x, y, z])
# to do: make a dictionary of file names
# to do: write a function that will read in a single field given 
#        the field name by matching to a dictionary for the file name

# Import packages
import numpy as np

# File names and paths
path_n = '/mnt/scratch-lustre/morrison/lat_bin_data/' #'/mnt/raid-cita/morrison/bin_lat_data/'#  #'../lattice-dev-master/Pseudospec/openmp_dev/'#
path_bl = 'ri_128_4_001/'#''#
path = ['ri_128_4_001/'] #['ri_128_2_302/','ri_128_2_304/','ri_128_2_306/','ri_128_2_308/','ri_128_2_201/','ri_128_2_101/','ri_128_2_102/','ri_128_2_103/','ri_128_2_104/', 'ri_128_2_108/','ri_128_2_112/','ri_128_2_116/','ri_128_2_120/', 'ri_128_2_124/','ri_128_2_128/','ri_128_2_132/', 'ri_128_2_401/','ri_128_2_402/','ri_128_2_403/','ri_128_2_404/']#['ri_128_2_002/']# ['']#
run_ident_bl = '_128_4_001'#'_TESTING32_'#
run_ident = ['_128_4_001']#['_128_2_302', '_128_2_304', '_128_2_306', '_128_2_308', '_128_2_201', '_128_2_101', '_128_2_102', '_128_2_103', '_128_2_104', '_128_2_108','_128_2_112','_128_2_116','_128_2_120','_128_2_124', '_128_2_128', '_128_2_132', '_128_2_401', '_128_2_402', '_128_2_403', '_128_2_404'] #['_128_2_002']#['_TESTING32_DV_']#

en_bl_f = 'energy_spec' + run_ident_bl + '.out'  #'energy_spec_TESTING32_.out'   # Baseline run lattice averaged quantities
en_f = ['energy_spec']*len(run_ident)  #['energy_spec_TESTING32_DV_.out'] # lattice averaged quatities

zm_f = 'zeta_moments'

phi_bl_f = 'phi_lat' + run_ident_bl + '.out'      
phi_f = ['phi_lat']*len(run_ident)
dphi_bl_f = 'dphi_lat' + run_ident_bl + '.out'
dphi_f = ['dphi_lat']*len(run_ident)
chi_bl_f = 'chi_lat' + run_ident_bl + '.out'
chi_f = ['chi_lat']*len(run_ident)
dchi_bl_f = 'dchi_lat' + run_ident_bl + '.out'
dchi_f = ['dchi_lat']*len(run_ident)
zeta_bl_f = 'zeta_lat' + run_ident_bl + '.out'
zeta_f = ['zeta_lat']*len(run_ident)

lap_phi_bl_f = 'lap_phi_lat' + run_ident_bl + '.out'      
lap_phi_f = ['lap_phi_lat']*len(run_ident)
lap_dphi_bl_f = 'lap_dphi_lat' + run_ident_bl + '.out'
lap_dphi_f = ['lap_dphi_lat']*len(run_ident)
lap_chi_bl_f = 'lap_chi_lat' + run_ident_bl + '.out'
lap_chi_f = ['lap_chi_lat']*len(run_ident)
lap_dchi_bl_f = 'lap_dchi_lat' + run_ident_bl + '.out'
lap_dchi_f = ['lap_dchi_lat']*len(run_ident)
lap_zeta_bl_f = 'lap_zeta_lat' + run_ident_bl + '.out'
lap_zeta_f = ['lap_zeta_lat']*len(run_ident)

g2_phi_bl_f = 'g2_phi_lat' + run_ident_bl + '.out'      
g2_phi_f = ['g2_phi_lat']*len(run_ident)
g2_dphi_bl_f = 'g2_dphi_lat' + run_ident_bl + '.out'
g2_dphi_f = ['g2_dphi_lat']*len(run_ident)
g2_chi_bl_f = 'g2_chi_lat' + run_ident_bl + '.out'
g2_chi_f = ['g2_chi_lat']*len(run_ident)
g2_dchi_bl_f = 'g2_dchi_lat' + run_ident_bl + '.out'
g2_dchi_f = ['g2_dchi_lat']*len(run_ident)
g2_zeta_bl_f = 'g2_zeta_lat' + run_ident_bl + '.out'
g2_zeta_f = ['g2_zeta_lat']*len(run_ident)

dz_gdg_phi_bl_f = 'dzeta_gdg_phi_lat' + run_ident_bl + '.out'
dz_gdg_chi_bl_f = 'dzeta_gdg_chi_lat' + run_ident_bl + '.out'
dz_lap_phi_bl_f = 'dzeta_lap_phi_lat' + run_ident_bl + '.out'
dz_lap_chi_bl_f = 'dzeta_lap_chi_lat' + run_ident_bl + '.out'
dz_gdg_phi_f = ['dzeta_gdg_phi_lat']*len(run_ident)
dz_gdg_chi_f = ['dzeta_gdg_chi_lat']*len(run_ident)
dz_lap_phi_f = ['dzeta_lap_phi_lat']*len(run_ident)
dz_lap_chi_f = ['dzeta_lap_chi_lat']*len(run_ident)

spec_bl_f = 'spectrum_pp' + run_ident_bl +'.out'
spec_f = ['spectrum_pp']*len(run_ident)

for i in range(0,len(run_ident)):
	en_f[i] = en_f[i] + run_ident[i] + '.out'
	phi_f[i] = phi_f[i] + run_ident[i] + '.out'
	dphi_f[i] = dphi_f[i] + run_ident[i] + '.out'
	chi_f[i] = chi_f[i] + run_ident[i] + '.out'
	dchi_f[i] = dchi_f[i] + run_ident[i] + '.out'
	zeta_f[i] = zeta_f[i] + run_ident[i] + '.out'
	lap_phi_f[i] = lap_phi_f[i] + run_ident[i] + '.out'
	lap_dphi_f[i] = lap_dphi_f[i] + run_ident[i] + '.out'
	lap_chi_f[i] = lap_chi_f[i] + run_ident[i] + '.out'
	lap_dchi_f[i] = lap_dchi_f[i] + run_ident[i] + '.out'
	lap_zeta_f[i] = lap_zeta_f[i] + run_ident[i] + '.out'
	g2_phi_f[i] = g2_phi_f[i] + run_ident[i] + '.out'
	g2_dphi_f[i] = g2_dphi_f[i] + run_ident[i] + '.out'
	g2_chi_f[i] = g2_chi_f[i] + run_ident[i] + '.out'
	g2_dchi_f[i] = g2_dchi_f[i] + run_ident[i] + '.out'
	g2_zeta_f[i] = g2_zeta_f[i] + run_ident[i] + '.out'
	spec_f[i] = spec_f[i] + run_ident[i] + '.out'
	dz_gdg_phi_f[i] = dz_gdg_phi_f[i] + run_ident[i] + '.out'
	dz_gdg_chi_f[i] = dz_gdg_chi_f[i] + run_ident[i] + '.out'
	dz_lap_phi_f[i] = dz_lap_phi_f[i] + run_ident[i] + '.out'
	dz_lap_chi_f[i] = dz_lap_chi_f[i] + run_ident[i] + '.out'

# Dictionary for file names
#file_dict = {'phi':phi_f[0], 'chi':chi_f[0], 'dphi':dphi[0], 'dchi':dchi[0], 'lap_phi':lap_phi[0]}

print('Plotting runs: ', run_ident)

# Lattice parameters
nx = 128; ny = 128; nz = 128
nlat = nx*ny*nz
llen = 0.875 #1.75
dx = llen/nx
dk = 2*np.pi/llen
nfld = 2

# Potential parameters
phi_p = 8.5
phi_w = 0.1
m2_p = -1600.
lambda_chi = 1.6e5
POTOPT = 4

# Data outputting parameters
sl = 3*2**2#256+128#2**2#  # steplat
ss = 2**0  # stepspec
ds = 2**3  # down sampling of lattice to plot
ne = 257+128 #257   # number of energy output steps
nl = (ne-1)//sl+1  # number of lattice output steps
ns = nl//ss        # number of spectra output steps            
nsk = int(np.sqrt(((nx/2+1)**2+(ny/2+1)**2+(nz/2+1)**2)+1.))  # number of bins in the spectra
k_ar = dk*np.arange(0,nsk)  # array of k values binned in spec 
k_nyq = dk*(nx//2+1)

# Indexing constants
# en indexing
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; rhoGrav_i = 6; rhoCons_i = 7; hub_i = 8; phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12
# zm indexing
m1 = 0; m2 = 1; m3 = 2; m4 = 3

# Colours
c_ar = ['b','k','r','g','y','c']
ls_ar = ['-','--','-.']

# File reading functions

# Function to read energies and mean fields from file
# en_bl Formatted [time, column]
# en Formatted [run, time, column]
def load_energy():
	# Load baseline energies
	en_bl = np.loadtxt(path_n + path_bl + en_bl_f)
	en_bl[:,hub_i] = np.sqrt(-en_bl[:,hub_i]/3.)
	# Load enegries from additional runs
	en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
	for i in range(0,len(en_f)):
		en[i] = np.loadtxt(path_n + path[i] + en_f[i])
		en[i,:,hub_i] = np.sqrt(-en[i,:,hub_i]/3.)
	return en_bl, en

# Function to read \zeta moments from file
# zm_bl: Formatted [time, column]
# zm: Formatted [run, time, column]
def load_moments():
	# Load baseline \zeta moments
	zm_bl = np.loadtxt(path_n + path_bl + zm_f + run_ident_bl + '.out')
	# Load \zeta moments from additional runs
	zm = np.zeros((np.shape(run_ident) + np.shape(zm_bl)))
	for i in range(0,len(run_ident)):
		zm[i] = np.loadtxt(path_n + path[i] + zm_f + run_ident[i] + '.out')
	return zm_bl, zm

# Function to read in energies and lattice averages from a single file
def load_ener(f_in, path_in):
        en = np.loadtxt(path_n + path_in + f_in)
        en[:,hub_i] = np.sqrt(-en[:,hub_i]/3.)
        return en

# Function to read in a lattice variable from file and resize the array
def load_lat(f_in, path_in):
	lat = np.fromfile(path_n + path_in + f_in, dtype=np.double, count=-1)
	lat = np.resize(lat, (nl,nlat))
	return lat

# Function to read spectra from file and resize arrays
# to do: check ordering of resize
# to do: the resizing is bugged/transposing
#def load_spec(f_in, path_in):
#	spec = np.fromfile(path_n + path_in + f_in, dtype=np.double, count=-1)
#	print('np.shape(spec) = ', np.shape(spec))
#	spec = np.resize(spec, (ns,4*nfld,nsk))
#	print('np.shape(spec) = ', np.shape(spec))
#	spec = np.transpose(spec, (1,0,2))
#	return spec


# Function to read spectra from file and resize arrays
# to do: check ordering of resize
# to do: the resizing is bugged/transposing
def load_spec(f_in, path_in):
	spec = np.fromfile(path_n + path_in + f_in, dtype=np.double, count=-1)
	print('np.shape(spec) = ', np.shape(spec))
	spec = np.resize(spec, (ns,2*nfld,2*nfld,nsk))
	print('np.shape(spec) = ', np.shape(spec))
	spec = np.transpose(spec, (2,1,0,3))
	return spec

# Function to read fields and momenta from file and resize the arrays
def load_fld():
	# Load baseline fields
	phi_bl = np.fromfile(path_n + path_bl + phi_bl_f, dtype=np.double , count=-1)
	dphi_bl = np.fromfile(path_n + path_bl + dphi_bl_f, dtype=np.double , count=-1)
	if (chi_bl_f != ''):
		chi_bl = np.fromfile(path_n + path_bl + chi_bl_f, dtype=np.double , count=-1)
		dchi_bl = np.fromfile(path_n + path_bl + dchi_bl_f, dtype=np.double , count=-1)
	# Load fields from additional runs
	phi = np.zeros((np.shape(phi_f) + np.shape(phi_bl)))
	dphi = np.zeros((np.shape(dphi_f) + np.shape(dphi_bl)))
	for i in range(0,len(phi_f)):
		phi[i] = np.fromfile(path_n + path[i] + phi_f[i], dtype=np.double , count=-1)
		dphi[i] = np.fromfile(path_n + path[i] + dphi_f[i], dtype=np.double , count=-1)
	if (chi_bl_f != ''):
		chi = np.zeros((np.shape(chi_f) + np.shape(chi_bl)))
		dchi = np.zeros((np.shape(dchi_f) + np.shape(chi_bl)))
		for i in range(0,len(chi_f)):
			chi[i] = np.fromfile(path_n + path[i] + chi_f[i], dtype=np.double , count=-1)
			dchi[i] = np.fromfile(path_n + path[i] + dchi_f[i], dtype=np.double , count=-1)
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

# Function to read laplacian of fields and momenta from file and resize the arrays
def load_lap_fld():
	# Load baseline laplacian of fields
	lap_phi_bl = np.fromfile(path_n + path_bl + lap_phi_bl_f, dtype=np.double , count=-1)
	lap_dphi_bl = np.fromfile(path_n + path_bl + lap_dphi_bl_f, dtype=np.double , count=-1)
	if (lap_chi_bl_f != ''):
		lap_chi_bl = np.fromfile(path_n + path_bl + lap_chi_bl_f, dtype=np.double , count=-1)
		lap_dchi_bl = np.fromfile(path_n + path_bl + lap_dchi_bl_f, dtype=np.double , count=-1)
	# Load fields from additional runs
	lap_phi = np.zeros((np.shape(lap_phi_f) + np.shape(lap_phi_bl)))
	lap_dphi = np.zeros((np.shape(lap_dphi_f) + np.shape(lap_dphi_bl)))
	for i in range(0,len(lap_phi_f)):
		lap_phi[i] = np.fromfile(path_n + path[i] + lap_phi_f[i], dtype=np.double , count=-1)
		lap_dphi[i] = np.fromfile(path_n + path[i] + lap_dphi_f[i], dtype=np.double , count=-1)
	if (lap_chi_bl_f != ''):
		lap_chi = np.zeros((np.shape(lap_chi_f) + np.shape(lap_chi_bl)))
		lap_dchi = np.zeros((np.shape(lap_dchi_f) + np.shape(lap_chi_bl)))
		for i in range(0,len(lap_chi_f)):
			lap_chi[i] = np.fromfile(path_n + path[i] + lap_chi_f[i], dtype=np.double , count=-1)
			lap_dchi[i] = np.fromfile(path_n + path[i] + lap_dchi_f[i], dtype=np.double , count=-1)
	# Resize data
	lap_phi_bl = np.resize(lap_phi_bl,(nl,nlat))      # Formatted [time, lat site]
	lap_dphi_bl = np.resize(lap_dphi_bl,(nl,nlat))    # Formatted [time, lat site]
	if (lap_chi_bl_f != ''):
		lap_chi_bl = np.resize(lap_chi_bl,(nl,nlat))    # Formatted [time, lat site]
		lap_dchi_bl = np.resize(lap_dchi_bl,(nl,nlat))  # Formatted [time, lat site]
	if (len(lap_phi_f) > 0):
		lap_phi = np.resize(lap_phi, (len(lap_phi_f),nl,nlat))     # Formatted [run, time, lat site]
		lap_dphi = np.resize(lap_dphi, (len(lap_dphi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (len(lap_chi_f) > 0):
		lap_chi = np.resize(lap_chi, (len(lap_chi_f),nl,nlat))     # Formatted [run, time, lat site]
		lap_dchi = np.resize(lap_dchi, (len(lap_dchi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (lap_chi_bl_f == ''):
		return lap_phi_bl, lap_dphi_bl, lap_phi, lap_dphi, 0, 0, 0, 0
	else:
		return lap_phi_bl, lap_dphi_bl, lap_chi_bl, lap_dchi_bl, lap_phi, lap_dphi, lap_chi, lap_dchi

# Function to read grad squared of fields and momenta from file and resize the arrays
def load_g2_fld():
	# Load baseline laplacian of fields
	f1_bl = np.fromfile(path_n + path_bl + g2_phi_bl_f, dtype=np.double , count=-1)
	df1_bl = np.fromfile(path_n + path_bl + g2_dphi_bl_f, dtype=np.double , count=-1)
	if (lap_chi_bl_f != ''):
		f2_bl = np.fromfile(path_n + path_bl + g2_chi_bl_f, dtype=np.double , count=-1)
		df2_bl = np.fromfile(path_n + path_bl + g2_dchi_bl_f, dtype=np.double , count=-1)
	# Load fields from additional runs
	f1 = np.zeros((np.shape(g2_phi_f) + np.shape(f1_bl)))
	df1 = np.zeros((np.shape(g2_dphi_f) + np.shape(f1_bl)))
	for i in range(0,len(g2_phi_f)):
		f1[i] = np.fromfile(path_n + path[i] + g2_phi_f[i], dtype=np.double , count=-1)
		df1[i] = np.fromfile(path_n + path[i] + g2_dphi_f[i], dtype=np.double , count=-1)
	if (g2_chi_bl_f != ''):
		f2 = np.zeros((np.shape(g2_chi_f) + np.shape(f1_bl)))
		df2 = np.zeros((np.shape(g2_dchi_f) + np.shape(f1_bl)))
		for i in range(0,len(g2_chi_f)):
			f2[i] = np.fromfile(path_n + path[i] + g2_chi_f[i], dtype=np.double , count=-1)
			df2[i] = np.fromfile(path_n + path[i] + g2_dchi_f[i], dtype=np.double , count=-1)
	# Resize data
	f1_bl = np.resize(f1_bl,(nl,nlat))      # Formatted [time, lat site]
	df1_bl = np.resize(df1_bl,(nl,nlat))    # Formatted [time, lat site]
	if (g2_chi_bl_f != ''):
		f2_bl = np.resize(f2_bl,(nl,nlat))    # Formatted [time, lat site]
		df2_bl = np.resize(df2_bl,(nl,nlat))  # Formatted [time, lat site]
	if (len(g2_phi_f) > 0):
		f1 = np.resize(f1, (len(g2_phi_f),nl,nlat))     # Formatted [run, time, lat site]
		df1 = np.resize(df1, (len(g2_dphi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (len(g2_chi_f) > 0):
		f2 = np.resize(f2, (len(g2_chi_f),nl,nlat))     # Formatted [run, time, lat site]
		df2 = np.resize(df2, (len(g2_dchi_f),nl,nlat))  # Formatted [run, time, lat site]
	if (chi_bl_f == ''):
		return f1_bl, df1_bl, f1, df1, 0, 0, 0, 0
	else:
		return f1_bl, df1_bl, f2_bl, df2_bl, f1, df1, f2, df2

# Function to read zeta from file and resize the arrays
def load_zeta(zeta_zero=True):
	zeta_bl = np.fromfile(path_n + path_bl+ zeta_bl_f, dtype=np.double , count=-1)
	zeta = np.zeros((np.shape(zeta_f) + np.shape(zeta_bl)))
	for i in range(0,len(zeta_f)):
		zeta[i] = np.fromfile(path_n + path[i] + zeta_f[i], dtype=np.double , count=-1)
	# resize data
	zeta_bl = np.resize(zeta_bl,(nl,nlat))      # Formatted [time, lat site]
	if (len(zeta_f) > 0):
		zeta = np.resize(zeta, (len(zeta_f),nl,nlat))     # Formatted [run, time, lat site]
	if zeta_zero:
		zeta_bl = zeta_bl - zeta_bl[0]
		for i in range(0,len(zeta)):
			zeta[i] = zeta[i] - zeta[i,0]
	return zeta_bl, zeta

# Function to read laplacian zeta from file and resize the arrays
def load_lap_zeta(zeta_zero=True):
	z_bl = np.fromfile(path_n + path_bl+ lap_zeta_bl_f, dtype=np.double , count=-1)
	z = np.zeros((np.shape(lap_zeta_f) + np.shape(z_bl)))
	for i in range(0,len(zeta_f)):
		z[i] = np.fromfile(path_n + path[i] + lap_zeta_f[i], dtype=np.double , count=-1)
	# resize data
	z_bl = np.resize(z_bl,(nl,nlat))      # Formatted [time, lat site]
	if (len(zeta_f) > 0):
		z = np.resize(z, (len(zeta_f),nl,nlat))     # Formatted [run, time, lat site]
	if zeta_zero:
		z_bl = z_bl - z_bl[0]
		for i in range(0,len(zeta)):
			z[i] = z[i] - z[i,0]
	return z_bl, z

# Function to load the partial contibutions to dzeta
def load_dzeta_part():
	dz_lapp_bl = np.fromfile(path_n + path_bl + dz_lap_phi_bl_f, dtype=np.double, count=-1)
	dz_gdgp_bl = np.fromfile(path_n + path_bl + dz_gdg_phi_bl_f, dtype=np.double, count=-1)
	dz_lapp = np.zeros((np.shape(run_ident) + np.shape(dz_lapp_bl)))
	dz_gdgp = np.zeros((np.shape(run_ident) + np.shape(dz_gdgp_bl)))
	for i in range(0,len(run_ident)):
		dz_lapp[i] = np.fromfile(path_n + path[i] + dz_lap_phi_f[i], dtype=np.double, count=-1)
		dz_gdgp[i] = np.fromfile(path_n + path[i] + dz_gdg_phi_f[i], dtype=np.double, count=-1)
	dz_lapp_bl = np.resize(dz_lapp_bl,(nl,nlat))
	dz_gdgp_bl = np.resize(dz_gdgp_bl,(nl,nlat))
	dz_lapp = np.resize(dz_lapp,(len(run_ident),nl,nlat))
	dz_gdgp = np.resize(dz_gdgp,(len(run_ident),nl,nlat))
	if (nfld>1):
		dz_lapc_bl = np.fromfile(path_n + path_bl + dz_lap_chi_bl_f, dtype=np.double, count=-1)
		dz_gdgc_bl = np.fromfile(path_n + path_bl + dz_gdg_chi_bl_f, dtype=np.double, count=-1)
		dz_lapc = np.zeros((np.shape(run_ident) + np.shape(dz_lapc_bl)))
		dz_gdgc = np.zeros((np.shape(run_ident) + np.shape(dz_gdgc_bl)))
		for i in range(0,len(run_ident)):
			dz_lapc[i] = np.fromfile(path_n + path[i] + dz_lap_chi_f[i], dtype=np.double, count=-1)
			dz_gdgc[i] = np.fromfile(path_n + path[i] + dz_gdg_chi_f[i], dtype=np.double, count=-1)
		dz_lapc_bl = np.resize(dz_lapc_bl,(nl,nlat))
		dz_gdgc_bl = np.resize(dz_gdgc_bl,(nl,nlat))
		dz_lapc = np.resize(dz_lapc,(len(run_ident),nl,nlat))
		dz_gdgc = np.resize(dz_gdgc,(len(run_ident),nl,nlat))
	return dz_lapp_bl, dz_gdgp_bl, dz_lapc_bl, dz_gdgc_bl, dz_lapp, dz_gdgp, dz_lapc, dz_gdgc

# Function to read spectra from file and resize arrays
# to do: check ordering of resize
# to do: the resizing is bugged/transposing
def load_spec2():
	spec_bl = np.fromfile(path_n + path_bl + spec_bl_f, dtype=np.double, count=-1)
	spec = np.zeros((np.shape(run_ident)) + np.shape(spec_bl))
	for i in range(0,len(run_ident)):
		spec[i] = np.fromfile(path_n + path[i] + spec_f[i], dtype=np.double, count=-1)
		# resize data
	print('np.shape(spec) = ', np.shape(spec))
	spec_bl = np.resize(spec_bl, (ns,nsk,4*nfld))
	spec_bl = np.transpose(spec_bl, (0,2,1))
	if (len(run_ident)>0):
		#spec = np.resize(spec, (len(run_ident),ns,nsk,4*nfld))
		spec = np.resize(spec, (len(run_ident),ns,4*nfld,nsk))
		print('np.shape(spec) = ', np.shape(spec))
		#spec = np.transpose(spec, (0,3,1,2))  # Formatted [run, component, t, k]
		spec = np.transpose(spec, (0,2,1,3))  # Formatted [run, component, t, k]
		print('np.shape(spec) = ', np.shape(spec))
	return spec_bl, spec







