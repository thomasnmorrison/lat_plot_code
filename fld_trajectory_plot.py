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
en_bl_f = 'energy_spec_TESTING32_.out' # Baseline run lattice averaged quantities
en_f = ['energy_spec_TESTING32_DV_.out'] # lattice averaged quatities
phi_bl_f = 'phi_lat_TESTING32_.out'
phi_f = ['phi_lat_TESTING32_DV_.out']
dphi_bl_f = 'dphi_lat_TESTING32_.out'
dphi_f = ['dphi_lat_TESTING32_DV_.out']
chi_bl_f = 'chi_lat_TESTING32_.out'
chi_f = ['chi_lat_TESTING32_DV_.out']
dchi_bl_f = 'dchi_lat_TESTING32_.out'
dchi_f = ['dchi_lat_TESTING32_DV_.out']

# Run parameters
nx = 32; ny = 32; nz = 32
sl = 2**2 # steplat
ds = 2**6 # down sampling of lattice to plot

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
	en[i] = np.loadtxt(path_n+en_f[i])
	phi[i] = np.fromfile(path_n+phi_f[i], dtype=np.double , count=-1)
	dphi[i] = np.fromfile(path_n+dphi_f[i], dtype=np.double , count=-1)

if (chi_bl_f != ''):
	chi_bl = np.fromfile(path_n+chi_bl_f, dtype=np.double , count=-1)
	dchi_bl = np.fromfile(path_n+dchi_bl_f, dtype=np.double , count=-1)
	chi = np.zeros((np.shape(chi_f) + np.shape(chi_bl)))
	dchi = np.zeros((np.shape(dchi_f) + np.shape(chi_bl)))
	for i in range(0,len(chi_f)):
		chi[i] = np.fromfile(path_n+chi_f[i], dtype=np.double , count=-1)
		dchi[i] = np.fromfile(path_n+dchi_f[i], dtype=np.double , count=-1)
 
# Reshape lattice data
print('np.shape(phi_bl) = ', np.shape(phi_bl))
print('np.shape(en_bl) = ', np.shape(en_bl))
print('np.shape(phi) = ', np.shape(phi))
print('np.shape(en) = ', np.shape(en))
nlat = nx*ny*nz
t =  int(len(en_bl[:,0]-1)/sl) + 1
print('t = ', t)
phi_bl = np.resize(phi_bl,(t,nlat)); phi_bl = phi_bl.transpose()       # Formatted [lat site, time]
dphi_bl = np.resize(dphi_bl,(t,nlat)); dphi_bl = dphi_bl.transpose()   # Formatted [lat site, time]
if (chi_bl_f != ''):
	chi_bl = np.resize(chi_bl,(t,nlat)); chi_bl = chi_bl.transpose()     # Formatted [lat site, time]
	dchi_bl = np.resize(dchi_bl,(t,nlat)); dchi_bl = dchi_bl.transpose() # Formatted [lat site, time]

if (len(phi_f) > 0):
	phi = np.resize(phi, (len(phi_f),t,nlat)); phi = phi.transpose((0,2,1))       # Formatted [run, lat site, time]
	dphi = np.resize(dphi, (len(dphi_f),t,nlat)); dphi = dphi.transpose((0,2,1))  # Formatted [run, lat site, time]
if (len(chi_f) > 0):
	chi = np.resize(chi, (len(chi_f),t,nlat)); chi = chi.transpose((0,2,1))       # Formatted [run, lat site, time]
	dchi = np.resize(dchi, (len(dchi_f),t,nlat)); dchi = dchi.transpose((0,2,1))  # Formatted [run, lat site, time]


print('np.shape(phi_bl) = ', np.shape(phi_bl))
print('np.shape(phi) = ', np.shape(phi))
print('np.shape(en) = ', np.shape(en))


# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8
phi_i = 9; dphi_i = 10; chi_i = 11; dchi_i = 12

# Make plots
nfig = 0

# Plot 1: \delta\phi clocked on \alpha, \Delta\phi clocked on \alpha
nfig += 1
print('In plot: ', nfig)
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = [r'$\delta\phi$',r'$\Delta\phi$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$M_{Pl}^{-1}$', r'$M_{Pl}^{-1}$']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

ax[0].plot(np.log(en_bl[::sl,a_i]), (phi_bl[:,:]-en_bl[::sl,phi_i]).T)
for i in range(0,len(phi_f)):
        ax[0].plot(np.log(en[i,::sl,a_i]), (phi[i,::ds,:]-en[i,::sl,phi_i]).T)
        ax[1].plot(np.log(en_bl[::sl,a_i]), (phi[i,::ds,:]-phi_bl[::ds,:]).T)

fig.set_tight_layout(True)
if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()
        
# Plot 2: \delta\dot{\phi} clocked on \alpha, \Delta\dot{\phi} clocked on \alpha 
nfig += 1
print('In plot: ', nfig)
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r'$\phi-\bar{\phi}$'
s_title = [r'$\delta\dot{\phi}$', r'$\Delta\dot{\phi}$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$(m_\phi M_{Pl})^{-1}$', r'$(m_\phi M_{Pl})^{-1}$']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

ax[0].plot(np.log(en_bl[::sl,a_i]), ((dphi_bl[::ds,:]-en_bl[::sl,dphi_i])/en_bl[::sl,a_i]**3).T)
for i in range(0,len(dphi_f)):
	ax[0].plot(np.log(en[i,::sl,a_i]), ((dphi[i,::ds,:]-en[i,::sl,dphi_i])/en[i,::sl,a_i]**3).T)
	ax[1].plot(np.log(en_bl[::sl,a_i]), (dphi[i,::ds,:]/en[i,::sl,a_i]**3-dphi_bl[::ds,:]/en_bl[::sl,a_i]**3).T)

fig.set_tight_layout(True)
if (SCALE_FIG == True):
	fig.set_size_inches(FIG_SIZE)
if (SAVE_FIG[nfig-1] == True):
	fig.savefig()

if (chi_bl_f != ''):
# Plot 3: \delta\chi clocked on \phi, \delta\chi clocked on \alpha
	nfig += 1
	print('In plot: ', nfig)
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r'$\chi-\bar{\chi}$'
	s_title = [r'$\delta\chi$',r'$\Delta\chi$']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
	y_lab = [r'$M_{Pl}^{-1}$', r'$M_{Pl}^{-1}$']
	#fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_title(s_title[i], fontsize = stitle_fs)
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	ax[0].plot(np.log(en_bl[::sl,a_i]), (chi_bl[::ds,:]-en_bl[::sl,chi_i]).T)
	for i in range(0,len(chi_f)):
		ax[0].plot(np.log(en[i,::sl,a_i]), (chi[i,::ds,:]-en[i,::sl,chi_i]).T)
		ax[1].plot(np.log(en_bl[::sl,a_i]), (chi[i,::ds,:]-chi_bl[::ds,:]).T)

	fig.set_tight_layout(True)
	if (SCALE_FIG == True):
		fig.set_size_inches(FIG_SIZE)
	if (SAVE_FIG[nfig-1] == True):
		fig.savefig()

if (dchi_bl_f != ''):
# Plot 4: \delta\Pi_phi clocked on \phi, \delta\Pi_\phi clocked on \alpha
	nfig += 1
	print('In plot: ', nfig)
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r'$\chi-\bar{\chi}$'
	s_title = [r'$\delta\dot{\chi}$', r'$\Delta\dot{\chi}$']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
	y_lab = [r'$(m_\phi M_{Pl})^{-1}$', r'$(m_\phi M_{Pl})^{-1}$']
	#fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_title(s_title[i], fontsize = stitle_fs)
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	ax[0].plot(np.log(en_bl[::sl,a_i]), ((dchi_bl[::ds,:]-en_bl[::sl,dchi_i])/en_bl[::sl,a_i]**3).T)
	for i in range(0,len(dchi_f)):
		ax[0].plot(np.log(en[i,::sl,a_i]), ((dchi[i,::ds,:]-en[i,::sl,dchi_i])/en[i,::sl,a_i]**3).T)
		ax[1].plot(np.log(en_bl[::sl,a_i]), (dchi[i,::ds,:]/en[i,::sl,a_i]**3-dchi_bl[::ds,:]/en_bl[::sl,a_i]**3).T)

	fig.set_tight_layout(True)
	if (SCALE_FIG == True):
		fig.set_size_inches(FIG_SIZE)
	if (SAVE_FIG[nfig-1] == True):
		fig.savefig()

plt.show()
