# fld_real_space_plot.py

# Script to plot real space slices of the fields
# Plot 1: Colormap of \delta fields and \dot{fields} on a 2d slice at specified time slices
# Plot 2: Colormap of \Delta fields and \dot{fields} on a 2d slice at specified time slices
# Plot 3: Colormap of \zeta_{V_0}, \zeta, and \Delta\zeta on a 2nd slice at specified times

# to do: set colorbar scale

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = True
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

# Reshape lattice data
fld = np.reshape(fld, (2,nfld,nl,nx,ny,nz))
dfld = np.reshape(dfld, (2,nfld,nl,nx,ny,nz))
zeta_bl = np.reshape(zeta_bl, (nl,nx,ny,nz))
zeta = np.reshape(zeta, (-1,nl,nx,ny,nz))

# Slices
t_i = np.zeros(5,np.int64)  # time indicies of slices to show
t_i[1] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p + phi_w)))
t_i[2] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p)))
t_i[3] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p - phi_w)))
t_i[-1] = -1
t_i = [32,40,48,56,64]
print('t_i = ', t_i)
z_i = 32

# Set histogram limits
vf = np.zeros((nfld,2))
vdf = np.zeros((nfld,2))
vz = np.zeros(2)
vdzp = np.zeros((2*nfld,2))
for j in range(0,nfld):
	vf_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])#/(en[0,::sl,a_i]**2))
	vf_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])#/(en[0,::sl,a_i]**2))
	vf[j] = [-np.max([np.absolute(vf_m),np.absolute(vf_p)]),np.max([np.absolute(vf_m),np.absolute(vf_p)])]
	vdf_m = np.min((dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T - dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
	vdf_p = np.max((dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:].T - dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
	vdf[j] = [-np.max([np.absolute(vdf_m),np.absolute(vdf_p)]),np.max([np.absolute(vdf_m),np.absolute(vdf_p)])]
vz_m = np.min((zeta-zeta_bl).T)
vz_p = np.max((zeta-zeta_bl).T)
#vz = [-np.max([np.absolute(vz_m),np.absolute(vz_p)]),np.max([np.absolute(vz_m),np.absolute(vz_p)])]
vz = [vz_m,vz_p]

# Make plots
nfig = 0

# Plot 1: Colormap of \delta fields and \dot{fields} and \Delta\zeta
#nfig += 1
#nr = 1; nc = len(t_i)  # number of subplot rows and columns
#fig = plt.figure(nfig)
#fig_n = 'fld_real_space_plot' + str(nfig) + run_ident[0] + '.png'
#f_title = r'Realspace Slice of Field Fluctuations'
#s_title = ['','']
#x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
#y_lab = [r'$\delta\phi$', r'$\delta\dot{\phi}$',r'$\delta\chi$', r'$\delta\dot{\chi}$']
#c_lab = [r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$',r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$']
#fig.suptitle(f_title)

#for j in range(0,nfld):
#	ax = axg.ImageGrid(fig, 411+2*j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
#	v_m = np.min(np.transpose(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
#	v_p = np.max(np.transpose(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
#	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
#	for i in range(0,nc):
#		f_slice = ax[i].imshow(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
#		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
#		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
#		ax[i].set_yticks([]); ax[i].set_xticks([])
#	ax.cbar_axes[0].colorbar(f_slice)
#	ax[0].set_ylabel(y_lab[2*j])
#	ax.cbar_axes[0].yaxis.set_label_position('right')
#	ax.cbar_axes[0].set_ylabel(c_lab[2*j])
#	ax = axg.ImageGrid(fig, 411+2*j+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
#	v_m = np.min((np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
#	v_p = np.max((np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
#	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
#	for i in range(0,nc):
#		f_slice = ax[i].imshow((dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]])/en[0,t_i[i]*sl,a_i]**3, cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
#		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
#		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
#		ax[i].set_yticks([]); ax[i].set_xticks([])
#	ax.cbar_axes[0].colorbar(f_slice)
#	ax[0].set_ylabel(y_lab[2*j+1])
#	ax.cbar_axes[0].yaxis.set_label_position('right')
#	ax.cbar_axes[0].set_ylabel(c_lab[2*j+1])
#if SAVE_FIGS:
#	plt.savefig(fig_n)

# Plot 1: Colormap of \delta fields and \dot{fields} and \Delta\zeta
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
fig_n = 'fld_real_space_plot' + str(nfig) + run_ident[0] + '.png'
f_title = r'Realspace Slice of Field Fluctuations'
s_title = [r'']*nc
for i in range(0,nc):
	s_title[i] = r'$\mathrm{{ln}}(a)={0:.2f}$'.format(np.log(en[0,t_i[i]*sl,a_i]))
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\delta\phi$', r'$\delta\dot{\phi}$',r'$\delta\chi$', r'$\delta\dot{\chi}$', r'$\Delta\zeta$']
c_lab = [r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$',r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$', r'']
#fig.suptitle(f_title)

for j in range(0,nfld):
	ax = axg.ImageGrid(fig, 511+2*j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	v_m = np.min(np.transpose(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	v_p = np.max(np.transpose(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])
	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
	for i in range(0,nc):
		f_slice = ax[i].imshow(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]-fld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]], cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
		ax[i].set_yticks([]); ax[i].set_xticks([])
	ax.cbar_axes[0].colorbar(f_slice)
	ax[0].set_ylabel(y_lab[2*j])
	if (j==0):
		for k in range(0,nc):
			ax[k].set_title(s_title[k])
	ax.cbar_axes[0].yaxis.set_label_position('right')
	ax.cbar_axes[0].set_ylabel(c_lab[2*j])
	ax = axg.ImageGrid(fig, 511+2*j+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	v_m = np.min((np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
	v_p = np.max((np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],:],(1,2,3,0))-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],:])/en[0,::sl,a_i]**3)
	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
	for i in range(0,nc):
		f_slice = ax[i].imshow((dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]-dfld0[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i]])/en[0,t_i[i]*sl,a_i]**3, cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
		ax[i].set_yticks([]); ax[i].set_xticks([])
	ax.cbar_axes[0].colorbar(f_slice)
	ax[0].set_ylabel(y_lab[2*j+1])
	ax.cbar_axes[0].yaxis.set_label_position('right')
	ax.cbar_axes[0].set_ylabel(c_lab[2*j+1])
ax = axg.ImageGrid(fig, 511+4, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	f_slice = ax[i].imshow(zeta[0,t_i[i],:,:,z_i]-zeta_bl[t_i[i],:,:,z_i], cmap='viridis', aspect='equal', origin='lower', vmin=vz[0], vmax=vz[1])#, vmin=v[0], vmax=v[1])
	ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
	ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
	ax[i].set_yticks([]); ax[i].set_xticks([])
ax.cbar_axes[0].colorbar(f_slice)
ax[0].set_ylabel(y_lab[2*nfld])
ax.cbar_axes[0].yaxis.set_label_position('right')
ax.cbar_axes[0].set_ylabel(c_lab[2*nfld])

fig.set_size_inches(8,6.4)
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 2: Colormap of \Delta fields and \dot{fields}
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
fig_n = 'fld_real_space_plot' + str(nfig) + run_ident[0] + '.png'
f_title = r'Realspace Slice of Field Fluctuations'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\Delta\phi$', r'$\Delta\dot{\phi}$',r'$\Delta\chi$', r'$\Delta\dot{\chi}$']
c_lab = [r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$',r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$']
fig.suptitle(f_title)

v = [-1.2e1,1.2e1]
for j in range(0,nfld):
	ax = axg.ImageGrid(fig, 411+2*j, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	v_m = np.min(fld[fld_i[nfld+j,0],fld_i[nfld+j,1]]-fld[fld_i[j,0],fld_i[j,1]])
	v_p = np.max(fld[fld_i[nfld+j,0],fld_i[nfld+j,1]]-fld[fld_i[j,0],fld_i[j,1]])
	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
	for i in range(0,nc):
		f_slice = ax[i].imshow(fld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]-fld[fld_i[j,0],fld_i[j,1],t_i[i],:,:,z_i], cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
		ax[i].set_yticks([]); ax[i].set_xticks([])
	ax.cbar_axes[0].colorbar(f_slice)
	ax[0].set_ylabel(y_lab[2*j])
	ax.cbar_axes[0].yaxis.set_label_position('right')
	ax.cbar_axes[0].set_ylabel(c_lab[2*j])
	ax = axg.ImageGrid(fig, 411+2*j+1, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
	v_m = np.min(np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1]],(1,2,3,0))/en[0,::sl,a_i]**3-np.transpose(dfld[fld_i[j,0],fld_i[j,1]],(1,2,3,0))/en_bl[::sl,a_i]**3)
	v_p = np.max(np.transpose(dfld[fld_i[nfld+j,0],fld_i[nfld+j,1]],(1,2,3,0))/en[0,::sl,a_i]**3-np.transpose(dfld[fld_i[j,0],fld_i[j,1]],(1,2,3,0))/en_bl[::sl,a_i]**3)
	v = [-np.max([np.absolute(v_m),np.absolute(v_p)]),np.max([np.absolute(v_m),np.absolute(v_p)])]
	for i in range(0,nc):
		f_slice = ax[i].imshow((dfld[fld_i[nfld+j,0],fld_i[nfld+j,1],t_i[i],:,:,z_i]/en[0,t_i[i]*sl,a_i]**3-dfld[fld_i[j,0],fld_i[j,1],t_i[i],:,:,z_i]/en_bl[t_i[i]*sl,a_i]**3), cmap='viridis', aspect='equal', origin='lower', vmin=v[0], vmax=v[1])
		ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
		ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
		ax[i].set_yticks([]); ax[i].set_xticks([])
	ax.cbar_axes[0].colorbar(f_slice)
	ax[0].set_ylabel(y_lab[2*j+1])
	ax.cbar_axes[0].yaxis.set_label_position('right')
	ax.cbar_axes[0].set_ylabel(c_lab[2*j+1])
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 3: Colormap of \zeta_{V_0}, \zeta, and \Delta\zeta on a 2nd slice at specified times
nfig += 1
nr = 1; nc = len(t_i)  # number of subplot rows and columns
fig = plt.figure(nfig)
fig_n = 'fld_real_space_plot' + str(nfig) + run_ident[0] + '.png'
f_title = r'Realspace Slice of $\Delta\zeta$'
s_title = ['','']
x_lab = [r'$\mathrm{ln}(a)$', r'$\bar{\phi}M_{Pl}^{-1}$']
y_lab = [r'$\zeta_{\Delta V=0}$', r'$\zeta$',r'$\Delta\zeta$']
c_lab = [r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$',r'$M_{Pl}^{-1}$', r'$(m_\phi M_{Pl})^{-1}$']
fig.suptitle(f_title)

ax = axg.ImageGrid(fig, 311, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	f_slice = ax[i].imshow(zeta_bl[t_i[i],:,:,z_i], cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
	ax[i].plot([0,1./(dx*en_bl[t_i[i]*sl,a_i]*en_bl[t_i[i]*sl,hub_i])],[0,0], c='w')
	ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
	ax[i].set_yticks([]); ax[i].set_xticks([])
ax.cbar_axes[0].colorbar(f_slice)	
ax[0].set_ylabel(y_lab[0])
ax.cbar_axes[0].yaxis.set_label_position('right')
#ax.cbar_axes[0].set_ylabel(c_lab[2*j])
ax = axg.ImageGrid(fig, 312, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	f_slice = ax[i].imshow(zeta[0,t_i[i],:,:,z_i], cmap='viridis', aspect='equal', origin='lower')#, vmin=v[0], vmax=v[1])
	ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
	ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
	ax[i].set_yticks([]); ax[i].set_xticks([])
ax.cbar_axes[0].colorbar(f_slice)	
ax[0].set_ylabel(y_lab[1])
ax.cbar_axes[0].yaxis.set_label_position('right')
#ax.cbar_axes[0].set_ylabel(c_lab[2*j])
ax = axg.ImageGrid(fig, 313, nrows_ncols=(nr,nc), axes_pad=0.05, cbar_location='right', cbar_mode='single')
for i in range(0,nc):
	f_slice = ax[i].imshow(zeta[0,t_i[i],:,:,z_i]-zeta_bl[t_i[i],:,:,z_i], cmap='viridis', aspect='equal', origin='lower', vmin=vz[0], vmax=vz[1])#, vmin=v[0], vmax=v[1])
	#f_slice = ax[i].imshow(zeta[0,-1,:,:,z_i+i]-zeta_bl[-1,:,:,z_i+i], cmap='viridis', aspect='equal', origin='lower', vmin=vz[0], vmax=vz[1])#, vmin=v[0], vmax=v[1])
	ax[i].plot([0,1./(dx*en[0,t_i[i]*sl,a_i]*en[0,t_i[i]*sl,hub_i])],[0,0], c='w')
	ax[i].annotate(r'$(aH)^{-1}$',(0,0+0.01*nx), color='w')
	ax[i].set_yticks([]); ax[i].set_xticks([])
ax.cbar_axes[0].colorbar(f_slice)	
ax[0].set_ylabel(y_lab[2])
ax.cbar_axes[0].yaxis.set_label_position('right')
ax.cbar_axes[0].set_ylabel(c_lab[2*j])


if SAVE_FIGS:
	plt.savefig(fig_n)

if SHOW_FIGS:
	plt.show()
