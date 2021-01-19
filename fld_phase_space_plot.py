# fld_phase_space_plot.py

# Script to plot the phase space of fields and momenta
# Plot 1: Trajectories through the \phi \Pi_\phi space
# Plot 2: 2d histograms of time slices of \phi \Pi_\phi space
# Plot 3: Contour plots of time slices of \phi \Pi_\phi space

# to do: set ticks
# to do: set labels
# to do: transpose hist while plotting
# to do: make a video for these


# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import mpl_toolkits.axes_grid1 as axg
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read in data
en_bl, en = load_energy()
phi_bl, dphi_bl, chi_bl, dchi_bl, phi, dphi, chi, dchi = load_fld()
zeta_bl, zeta = load_zeta()

fld = [[phi_bl, chi_bl],[phi[0],chi[0]]]; fld = np.array(fld)
dfld = [[dphi_bl, dchi_bl],[dphi[0],dchi[0]]]; dfld = np.array(dfld)
fld0 = [[en_bl[::sl, phi_i],en_bl[::sl, chi_i]],[en[0,::sl, phi_i],en[0,::sl, chi_i]]]; fld0 = np.array(fld0)
dfld0 = [[en_bl[::sl, dphi_i],en_bl[::sl, dchi_i]],[en[0,::sl, dphi_i],en[0,::sl, dchi_i]]]; dfld0 = np.array(dfld0)
f1_bl_i = [0,0]; f2_bl_i = [0,1]; f1_i = [1,0]; f2_i = [1,1]
fld_i = np.array([f1_bl_i, f2_bl_i, f1_i, f2_i])

n_bins = 128

# Slices
t_i = np.zeros(5,np.int64)  # time indicies of slices to show
t_i[1] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p + phi_w)))
t_i[2] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p)))
t_i[3] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p - phi_w)))
t_i[-1] = -1

t_i = np.zeros(6,np.int64)  # time indicies of slices to show
t_i[0] = 40
t_i[1] = 45
t_i[2] = 50
t_i[3] = 55
t_i[4] = 60
t_i[-1] = -1

# Make plots
nfig = 0

# Plot 1: Histograms of (\delta\phi_A, \delta\Pi_\phi)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
#fig.set_figheight(4.8/2)

for j in range(0,nfld):
	vf_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vdf_p = np.max(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 211+j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 2: Histograms of (\Delta\phi_A, \Delta\Pi_\phi)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
#fig.set_figheight(4.8/2)

for j in range(0,nfld):
	vf_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-fld[fld_i[j,0],fld_i[j,1],:])
	vf_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-fld[fld_i[j,0],fld_i[j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-dfld[fld_i[j,0],fld_i[j,1],:])
	vdf_p = np.max(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-dfld[fld_i[j,0],fld_i[j,1],:])
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 211+j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-fld[fld_i[j,0],fld_i[j,1],t_i[i]], dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-dfld[fld_i[j,0],fld_i[j,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 3: Histograms of (\delta\phi_A, \delta\phi_B), (\delta\Pi_\phi_A, \delta\Pi_\phi_B)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
#fig.set_figheight(4.8/2)
# 
vf_m = np.min(fld[fld_i[nfld,0],fld_i[nfld,1],:].T-fld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf_p = np.max(fld[fld_i[nfld,0],fld_i[nfld,1],:].T-fld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
vdf_m = np.min(fld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf_p = np.max(fld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
ax = axg.ImageGrid(fig, 211, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	hist, xe, ye = np.histogram2d(fld[fld_i[nfld,0],fld_i[nfld,1],t_i[i]]-fld0[fld_i[nfld,0],fld_i[nfld,1],t_i[i]], fld[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]]-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
	f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
# 
vf_m = np.min(dfld[fld_i[nfld,0],fld_i[nfld,1],:].T-dfld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf_p = np.max(dfld[fld_i[nfld,0],fld_i[nfld,1],:].T-dfld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
vdf_m = np.min(dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf_p = np.max(dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
ax = axg.ImageGrid(fig, 211+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	hist, xe, ye = np.histogram2d(dfld[fld_i[nfld,0],fld_i[nfld,1],t_i[i]]-dfld0[fld_i[nfld,0],fld_i[nfld,1],t_i[i]], dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]]-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
	f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 4: Histograms of (\delta\phi_A, \delta\Pi_\phi_B)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
# 
vf_m = np.min(fld[fld_i[nfld,0],fld_i[nfld,1],:].T-fld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf_p = np.max(fld[fld_i[nfld,0],fld_i[nfld,1],:].T-fld0[fld_i[nfld,0],fld_i[nfld,1],:])
vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
vdf_m = np.min(dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf_p = np.max(dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
ax = axg.ImageGrid(fig, 211, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	hist, xe, ye = np.histogram2d(fld[fld_i[nfld,0],fld_i[nfld,1],t_i[i]]-fld0[fld_i[nfld,0],fld_i[nfld,1],t_i[i]], dfld[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]]-dfld0[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
	f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
# 
vf_m = np.min(fld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vf_p = np.max(fld[fld_i[nfld+1,0],fld_i[nfld+1,1],:].T-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],:])
vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
vdf_m = np.min(dfld[fld_i[nfld,0],fld_i[nfld,1],:].T-dfld0[fld_i[nfld,0],fld_i[nfld,1],:])
vdf_p = np.max(dfld[fld_i[nfld,0],fld_i[nfld,1],:].T-dfld0[fld_i[nfld,0],fld_i[nfld,1],:])
vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
ax = axg.ImageGrid(fig, 211+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	hist, xe, ye = np.histogram2d(fld[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]]-fld0[fld_i[nfld+1,0],fld_i[nfld+1,1],t_i[i]], dfld[fld_i[nfld,0],fld_i[nfld,1],t_i[i]]-dfld0[fld_i[nfld,0],fld_i[nfld,1],t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
	f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 5: Histograms of (\delta\phi_A, \Delta\zeta), (\delta\Pi_\phi_A, \Delta\zeta)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
#fig.set_figheight(4.8/2)

for j in range(0,nfld):
	vf_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(zeta[0]-zeta_bl)
	vdf_p = np.max(zeta[0]-zeta_bl)
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 411+2*j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], zeta[0,t_i[i]]-zeta_bl[t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
	vf_m = np.min(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf_p = np.max(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(zeta[0]-zeta_bl)
	vdf_p = np.max(zeta[0]-zeta_bl)
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 411+2*j+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], zeta[0,t_i[i]]-zeta_bl[t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 6: Histograms of (\Delta\phi_A, \Delta\zeta), (\Delta\Pi_\phi_A, \Delta\zeta)
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
f_title = r''
s_title = ['','']
x_lab = [r'']
y_lab = [r'']
c_lab = [r'']
fig.suptitle(f_title)
#fig.set_figheight(4.8/2)

for j in range(0,nfld):
	vf_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-fld[fld_i[j,0],fld_i[j,1],:])
	vf_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-fld[fld_i[j,0],fld_i[j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(zeta[0]-zeta_bl)
	vdf_p = np.max(zeta[0]-zeta_bl)
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 411+2*j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-fld[fld_i[j,0],fld_i[j,1],t_i[i]], zeta[0,t_i[i]]-zeta_bl[t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
	vdf_m = np.min(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-dfld[fld_i[j,0],fld_i[j,1],:])
	vdf_p = np.max(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:]-dfld[fld_i[j,0],fld_i[j,1],:])
	vf = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min(zeta[0]-zeta_bl)
	vdf_p = np.max(zeta[0]-zeta_bl)
	vdf = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
	ax = axg.ImageGrid(fig, 411+2*j+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	for i in range(0,nc):
		hist, xe, ye = np.histogram2d(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]]-dfld[fld_i[j,0],fld_i[j,1],t_i[i]], zeta[0,t_i[i]]-zeta_bl[t_i[i]], bins=n_bins, range=[vf,vdf], density=True)
		f_slice = ax[i].imshow(hist, cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])

# Plot 7: test of x,y axis

if SHOW_FIGS:
	plt.show()
