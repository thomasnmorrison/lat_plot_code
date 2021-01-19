# fld_trajectory_contour_plot.py

# Script to plot contours of the field and field momentum trajectories
# Plot 1: Contour plot of baseline \delta\phi and \delta\dot{\phi} clocked on \alpha
# Plot 2: Contour plot of baseline \delta\chi and \delta\dot{\chi} clocked on \alpha
# Plot 3: Contour plot of \delta\phi and \delta\dot{\phi} clocked on \alpha
# Plot 4: Contour plot of \delta\chi and \delta\dot{\chi} clocked on \alpha
# Plot 5: Contour plot of \Delta\phi and \Delta\dot{\phi} clocked on \alpha
# Plot 6: Contour plot of \Delta\chi and \Delta\dot{\chi} clocked on \alpha

# to do:

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../run_data/'#'../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_128_3_000.out' # Baseline run lattice averaged quantities
en_f = ['energy_spec_128_3_002.out'] # lattice averaged quatities
phi_bl_f = 'chi_lat_128_3_000.out'
phi_f = ['chi_lat_128_3_002.out']
dphi_bl_f = 'dchi_lat_128_3_000.out'
dphi_f = ['dchi_lat_128_3_002.out']
chi_bl_f = ''
chi_f = []
dchi_bl_f = ''
dchi_f = []

# Run parameters
nx = 128; ny = 128; nz = 128
sl = 2**2 # steplat
ds = int(nx*ny*nz/2**3) # down sampling of lattice to plot
cont = np.array([0., 0.05, 0.25, 0.75, 0.95, 1.]) # contour levels

# Configuration parameters
SAVE_FIG = [False, False, False, False]
SCALE_FIG = [False, False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False, False]

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14

c_ar = ['b','r','g','y']
# Plot output file names


# Read in data
en_bl = np.loadtxt(path_n+en_bl_f)  # Formatted [time, column]
print('reading phi')
phi_bl = np.fromfile(path_n+phi_bl_f, dtype=np.double , count=-1)
print('reading dphi')
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

# Calculate contours
phi_bl_c = np.zeros((len(cont), t))             # Formatted [contour, time]
dphi_bl_c = np.zeros((len(cont), t))            # Formatted [contour, time]
phi_c = np.zeros((len(phi_f),len(cont), t))     # Formatted [run, contour, time]
dphi_c = np.zeros((len(dphi_f),len(cont), t))   # Formatted [run, contour, time]
Delta_phi_c = np.zeros((len(phi_f),len(cont), t))     # Formatted [run, contour, time]
Delta_dphi_c = np.zeros((len(dphi_f),len(cont), t))   # Formatted [run, contour, time]

if (chi_bl_f != ''):
	chi_bl_c = np.zeros((len(cont), t))
	dchi_bl_c = np.zeros((len(cont), t))
	chi_c = np.zeros((len(chi_f),len(cont), t))
	dchi_c = np.zeros((len(dchi_f),len(cont), t))
	Delta_chi_c = np.zeros((len(chi_f),len(cont), t))
	Delta_dchi_c = np.zeros((len(dchi_f),len(cont), t))

print('sorting phi')
phi_sort = np.sort(phi_bl, axis=0)
print('sorting dphi')
dphi_sort = np.sort(dphi_bl, axis=0)
if (chi_bl_f != ''):
	chi_sort = np.sort(chi_bl, axis=0)
	dchi_sort = np.sort(dchi_bl, axis=0)
for i,c in enumerate(cont):
	cont_ind = int(c*(nlat-1))
	phi_bl_c[i,:] = phi_sort[cont_ind,:]
	dphi_bl_c[i,:] = dphi_sort[cont_ind,:]
	if (chi_bl_f != ''):
		chi_bl_c[i,:] = chi_sort[cont_ind,:]
		dchi_bl_c[i,:] = dchi_sort[cont_ind,:]

for j in range(0,len(phi_f)):
	phi_sort = np.sort(phi[j], axis=0)
	dphi_sort = np.sort(dphi[j], axis=0)
	Delta_phi_sort = np.sort(phi[j]-phi_bl, axis=0)
	Delta_dphi_sort = np.sort(dphi[j]-dphi_bl, axis=0)
	if (chi_bl_f != ''):
		chi_sort = np.sort(chi[j], axis=0)
		dchi_sort = np.sort(dchi[j], axis=0)
		Delta_chi_sort = np.sort(chi[j]-chi_bl, axis=0)
		Delta_dchi_sort = np.sort(dchi[j]-dchi_bl, axis=0)
	for i,c in enumerate(cont):
		cont_ind = int(c*(nlat-1))
		phi_c[j,i,:] = phi_sort[cont_ind,:]
		dphi_c[j,i,:] = dphi_sort[cont_ind,:]
		Delta_phi_c[j,i,:] = Delta_phi_sort[cont_ind,:]
		Delta_dphi_c[j,i,:] = Delta_dphi_sort[cont_ind,:]
		if (chi_bl_f != ''):
			chi_c[j,i,:] = chi_sort[cont_ind,:]
			dchi_c[j,i,:] = dchi_sort[cont_ind,:]
			Delta_chi_c[j,i,:] = Delta_chi_sort[cont_ind,:]
			Delta_dchi_c[j,i,:] = Delta_dchi_sort[cont_ind,:]

# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8
phi_i = 11; dphi_i = 12; chi_i = 11; dchi_i = 12
#phi_i = 9; dphi_i = 10

# Make plots
nfig = 0

# Plot 1: Contour plot of \delta\phi clocked on \alpha 
nfig += 1
print('In plot: ', nfig)
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r''
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'', r'']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

for i in range(0,int(len(cont)/2)):
	ax[0].fill_between(np.log(en_bl[::sl,a_i]), (phi_bl_c[i,:]-en_bl[::sl,phi_i]).T, (phi_bl_c[-1-i,:]-en_bl[::sl,phi_i]).T, alpha=0.25, color='b')
	ax[1].fill_between(np.log(en_bl[::sl,a_i]), ((dphi_bl_c[i,:]-en_bl[::sl,dphi_i])/en_bl[::sl,a_i]**3).T, ((dphi_bl_c[-1-i,:]-en_bl[::sl,dphi_i])/en_bl[::sl,a_i]**3).T, alpha=0.25, color='b')

ax[0].plot(np.log(en_bl[::sl,a_i]), (phi_bl[::ds,:]-en_bl[::sl,phi_i]).T, lw=1., alpha=0.5, color='k')
ax[1].plot(np.log(en_bl[::sl,a_i]), ((dphi_bl[::ds,:]-en_bl[::sl,dphi_i])/en_bl[::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')

# Plot 2: Contour plot of baseline \delta\chi and \delta\dot{\chi} clocked on \alpha
nfig += 1
if (chi_bl_f != ''):
	print('In plot: ', nfig)
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r''
	s_title = [r'',r'']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
	y_lab = [r'', r'']
	#fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_title(s_title[i], fontsize = stitle_fs)
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	for i in range(0,int(len(cont)/2)):
		ax[0].fill_between(np.log(en_bl[::sl,a_i]), (chi_bl_c[i,:]-en_bl[::sl,chi_i]).T, (chi_bl_c[-1-i,:]-en_bl[::sl,chi_i]).T, alpha=0.25, color='b')
		ax[1].fill_between(np.log(en_bl[::sl,a_i]), ((dchi_bl_c[i,:]-en_bl[::sl,dchi_i])/en_bl[::sl,a_i]**3).T, ((dchi_bl_c[-1-i,:]-en_bl[::sl,dchi_i])/en_bl[::sl,a_i]**3).T, alpha=0.25, color='b')

	ax[0].plot(np.log(en_bl[::sl,a_i]), (chi_bl[::ds,:]-en_bl[::sl,chi_i]).T, lw=1., alpha=0.5, color='k')
	ax[1].plot(np.log(en_bl[::sl,a_i]), ((dchi_bl[::ds,:]-en_bl[::sl,dchi_i])/en_bl[::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')

# Plot 3: Contour plot of \delta\phi and \delta\dot{\phi} clocked on \alpha
nfig += 1
print('In plot: ', nfig)
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r''
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'', r'']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

for j in range(0,len(phi_f)):
	for i in range(0,int(len(cont)/2)):
		ax[0].fill_between(np.log(en[j,::sl,a_i]), (phi_c[j,i,:]-en[j,::sl,phi_i]).T, (phi_c[j,-1-i,:]-en[j,::sl,phi_i]).T, alpha=0.25, color=c_ar[j])
		ax[1].fill_between(np.log(en[j,::sl,a_i]), ((dphi_c[j,i,:]-en[j,::sl,dphi_i])/en[j,::sl,a_i]**3).T, ((dphi_c[j,-1-i,:]-en[j,::sl,dphi_i])/en[j,::sl,a_i]**3).T, alpha=0.25, color=c_ar[j])

	ax[0].plot(np.log(en[j,::sl,a_i]), (phi[j,::ds,:]-en[j,::sl,phi_i]).T, lw=1., alpha=0.5, color='k')
	ax[1].plot(np.log(en[j,::sl,a_i]), ((dphi[j,::ds,:]-en[j,::sl,dphi_i])/en[j,::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')

# Plot 4: Contour plot of \delta\chi and \delta\dot{\chi} clocked on \alpha
nfig += 1
if (chi_bl_f != ''):
	print('In plot: ', nfig)
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r''
	s_title = [r'',r'']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
	y_lab = [r'', r'']
	#fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_title(s_title[i], fontsize = stitle_fs)
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	for j in range(0,len(phi_f)):
		for i in range(0,int(len(cont)/2)):
			ax[0].fill_between(np.log(en[j,::sl,a_i]), (chi_c[j,i,:]-en[j,::sl,chi_i]).T, (chi_c[j,-1-i,:]-en[j,::sl,chi_i]).T, alpha=0.25, color=c_ar[j])
			ax[1].fill_between(np.log(en[j,::sl,a_i]), ((dchi_c[j,i,:]-en[j,::sl,dchi_i])/en[j,::sl,a_i]**3).T, ((dchi_c[j,-1-i,:]-en[j,::sl,dchi_i])/en[j,::sl,a_i]**3).T, alpha=0.25, color=c_ar[j])

		ax[0].plot(np.log(en[j,::sl,a_i]), (chi[j,::ds,:]-en[j,::sl,chi_i]).T, lw=1., alpha=0.5, color='k')
		ax[1].plot(np.log(en[j,::sl,a_i]), ((dchi[j,::ds,:]-en[j,::sl,dchi_i])/en[j,::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')

# Plot 5: Contour plot of \Delta\phi and \Delta\dot{\phi} clocked on \alpha
nfig += 1
print('In plot: ', nfig)
fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
f_title = r''
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'', r'']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	ax[i].set_title(s_title[i], fontsize = stitle_fs)
	ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
	ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
	ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

for j in range(0,len(phi_f)):
	for i in range(0,int(len(cont)/2)):
		ax[0].fill_between(np.log(en[j,::sl,a_i]), (Delta_phi_c[j,i,:]).T, (Delta_phi_c[j,-1-i,:]).T, alpha=0.25, color=c_ar[j])
		ax[1].fill_between(np.log(en[j,::sl,a_i]), ((Delta_dphi_c[j,i,:])/en[j,::sl,a_i]**3).T, ((Delta_dphi_c[j,-1-i,:])/en[j,::sl,a_i]**3).T, alpha=0.25, color=c_ar[j])

	ax[0].plot(np.log(en[j,::sl,a_i]), (phi[j,::ds,:]-phi_bl[::ds,:]).T, lw=1., alpha=0.5, color='k')
	ax[1].plot(np.log(en[j,::sl,a_i]), ((dphi[j,::ds,:])/en[j,::sl,a_i]**3-(dphi_bl[::ds,:])/en_bl[::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')

# Plot 6: Contour plot of \Delta\chi and \Delta\dot{\chi} clocked on \alpha
nfig += 1
if (chi_bl_f != ''):
	print('In plot: ', nfig)
	fig, ax = plt.subplots(nrows=2 , ncols=1 , sharex=False)
	f_title = r''
	s_title = [r'',r'']
	x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
	y_lab = [r'', r'']
	#fig.suptitle(f_title, fontsize = title_fs)
	for i in range(0,2):
		ax[i].set_title(s_title[i], fontsize = stitle_fs)
		ax[i].set_xlabel(x_lab[i], fontsize = x_lab_fs)
		ax[i].set_ylabel(y_lab[i], fontsize = y_lab_fs)
		ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

	for j in range(0,len(phi_f)):
		for i in range(0,int(len(cont)/2)):
			ax[0].fill_between(np.log(en[j,::sl,a_i]), (Delta_chi_c[j,i,:]).T, (Delta_chi_c[j,-1-i,:]).T, alpha=0.25, color=c_ar[j])
			ax[1].fill_between(np.log(en[j,::sl,a_i]), ((Delta_dchi_c[j,i,:])/en[j,::sl,a_i]**3).T, ((Delta_dchi_c[j,-1-i,:])/en[j,::sl,a_i]**3).T, alpha=0.25, color=c_ar[j])

		ax[0].plot(np.log(en[j,::sl,a_i]), (chi[j,::ds,:]-chi_bl[::ds,:]).T, lw=1., alpha=0.5, color='k')
		ax[1].plot(np.log(en[j,::sl,a_i]), ((dchi[j,::ds,:])/en[j,::sl,a_i]**3-(dchi_bl[::ds,:])/en_bl[::sl,a_i]**3).T, lw=1., alpha=0.5, color='k')


plt.show()
