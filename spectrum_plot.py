# spectrum_plot.py

# Script to plot the spectra from a lattice sim
# Plot 1: k/aH vs (cross)spectra for \phi, \Pi_\phi at set times
# Plot 2: k/aH vs (cross)spectra for \chi, \Pi_\chi at set times
# Plot 3: k/aH vs (cross)spectra for \zeta, \dot{\zeta} at set times
# Plot 4: ln(a) vs (cross)spectra for \phi, \Pi_\phi at set k
# Plot 5: ln(a) vs (cross)spectra for \chi, \Pi_\chi at set k
# Plot 6: ln(a) vs (cross)spectra for \zeta, \dot{\zeta} at set k

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_TESTING_.out' # Baseline run lattice averaged quantities
en_f = [] # lattice averaged quatities
spec_bl_f = 'spectrum_TESTING_.out'
spec_f = []

# Run parameters
nx = 8; ny = 8; nz = 8
ss = 2**2 # stepspec


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
spec_bl = np.loadtxt(path_n+spec_bl_f)

en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
spec = np.zeros((np.shape(spec_f) + np.shape(spec_bl)))
for i in range(0,len(en_f)):
	en[i] = np.loadtxt(path_n+en_f[i])
	spec[i] = np.loadtxt(path_n+spec_f[i])

# Reshape spectrum data
ns = int(np.sqrt((nx/2+1)**2+(ny/2+1)**2+(nz/2+1)**2+1))
x,y = np.shape(spec_bl)
spec_bl = np.reshape(spec_bl,(int(x/ns),ns,y))    # Formatted [time, k, spec]
print('np.shape(spec_bl) = ', np.shape(spec_bl))

# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8; phi_i = 9
zz_i = 1; dzdz_i = 2; rdzz_i = 3; idzz_i = 4
ff_i = [5,9]; dfdf_i = [6,10]; rdff_i = [7,11]; idff_i = [8,12]

# Calc H, \dot{\phi}
en_bl[:,hub_i] = np.sqrt(-en_bl[:,hub_i]/3.)

# Calc spectrum normalization

# Make plots
nfig = 0

# Plot 1: (cross)spectra for \phi, \Pi_\phi
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = [r'$\langle |\phi|^2 \rangle$', r'$\langle |\Pi_\phi|^2 \rangle$',
           r'$\mathrm{Re}\langle \phi^*\Pi_\phi \rangle$', r'$\mathrm{Im}\langle \phi^*\Pi_\phi \rangle$']
x_lab = [r'$k/(aH)$']
y_lab = [r'$\langle |\phi|^2 \rangle$', r'$\langle |\Pi_\phi|^2 \rangle$',
           r'$\mathrm{Re}\langle \phi^*\Pi_\phi \rangle$', r'$\mathrm{Im}\langle \phi^*\Pi_\phi \rangle$']
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].set_title(s_title[2*i+j], fontsize = stitle_fs)
		ax[i,j].set_xlabel(x_lab[0], fontsize = x_lab_fs)
		#ax[i,j].set_ylabel(y_lab[2*i+j], fontsize = y_lab_fs)
		ax[i,j].plot(spec_bl[0,1:,0]/(en_bl[1*ss,a_i]*en_bl[1*ss,hub_i]), spec_bl[1,1:,5+2*i+j])

if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

# Plot 2: (cross)spectra for \chi, \Pi_\chi
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = [r'$\langle |\chi|^2 \rangle$', r'$\langle |\Pi_\chi|^2 \rangle$',
           r'$\mathrm{Re}\langle \chi^*\Pi_\chi \rangle$', r'$\mathrm{Im}\langle \chi^*\Pi_\chi \rangle$']
x_lab = [r'$k/(aH)$']
y_lab = [r'$\langle |\chi|^2 \rangle$', r'$\langle |\Pi_\chi|^2 \rangle$',
           r'$\mathrm{Re}\langle \chi^*\Pi_\chi \rangle$', r'$\mathrm{Im}\langle \chi^*\Pi_\chi \rangle$']
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].set_title(s_title[2*i+j], fontsize = stitle_fs)
		ax[i,j].set_xlabel(x_lab[0], fontsize = x_lab_fs)
		#ax[i,j].set_ylabel(y_lab[2*i+j], fontsize = y_lab_fs)
		ax[i,j].plot(spec_bl[0,1:,0]/(en_bl[1*ss,a_i]*en_bl[1*ss,hub_i]), spec_bl[1,1:,9+2*i+j])

if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

# Plot 3: (cross)spectra for \zeta, \dot{\zeta}
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = [r'$\langle |\zeta|^2 \rangle$', r'$\langle |\zeta_\tau|^2 \rangle$',
           r'$\mathrm{Re}\langle \zeta^*\zeta_\tau \rangle$', r'$\mathrm{Im}\langle \zeta^*\zeta_\tau \rangle$']
x_lab = [r'$k/(aH)$']
y_lab = [r'$\langle |\zeta|^2 \rangle$', r'$\langle |\dot{\zeta}|^2 \rangle$',
           r'$\mathrm{Re}\langle \zeta^*\dot{\zeta} \rangle$', r'$\mathrm{Im}\langle \zeta^*\dot{\zeta} \rangle$']
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].set_title(s_title[2*i+j], fontsize = stitle_fs)
		ax[i,j].set_xlabel(x_lab[0], fontsize = x_lab_fs)
		#ax[i,j].set_ylabel(y_lab[2*i+j], fontsize = y_lab_fs)
		ax[i,j].plot(spec_bl[0,1:,0]/(en_bl[1*ss,a_i]*en_bl[1*ss,hub_i]), spec_bl[1,1:,1+2*i+j])

# Plot 4: ln(a) vs (cross)spectra for \phi, \Pi_\phi at set k
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = [r'$\langle |\phi|^2 \rangle$', r'$\langle |\Pi_\phi|^2 \rangle$',
           r'$\mathrm{Re}\langle \phi^*\Pi_\phi \rangle$', r'$\mathrm{Im}\langle \phi^*\Pi_\phi \rangle$']
x_lab = [r'$ln(a)$']
y_lab = [r'$\langle |\phi|^2 \rangle$', r'$\langle |\Pi_\phi|^2 \rangle$',
           r'$\mathrm{Re}\langle \phi^*\Pi_\phi \rangle$', r'$\mathrm{Im}\langle \phi^*\Pi_\phi \rangle$']
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].set_title(s_title[2*i+j], fontsize = stitle_fs)
		ax[i,j].set_xlabel(x_lab[0], fontsize = x_lab_fs)
		ax[i,j].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
		#ax[i,j].set_ylabel(y_lab[2*i+j], fontsize = y_lab_fs)
		ax[i,j].plot(np.log(en_bl[::ss,a_i]), spec_bl[:,1,5+2*i+j])


plt.show()
