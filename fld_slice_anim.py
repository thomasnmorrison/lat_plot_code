# fld_slice_anim.py

# Script to make an animation of field fluctuations on a 2d slice of the lattice.

# Animation 1: 

# to do: make a zoom in animation
# to do: make an equivalent script that used VisIt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import potential as pot
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save/show figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read in data
en = load_ener(en_f[0], path[0])
phi = load_lat(phi_f[0], path[0])
chi = load_lat(chi_f[0], path[0])

# Set potential parameters
pot.init_param(phi_p, phi_w, m2_p, lambda_chi_in=lambda_chi, POTOPT_in=POTOPT)

# Calculate \phi fluctuations
phi = (phi.T - en[::sl,phi_i]).T

# Reshape data to lattice sites
phi = np.reshape(phi, (nl,nx,ny,nz))
chi = np.reshape(chi, (nl,nx,ny,nz))

# Make mesh
X = np.arange(0,nx)
Y = np.arange(0,ny)
X, Y = np.meshgrid(X, Y)
z_i = 25

# Animation
nfig = 0

# Point of view
elev = 45.
azim = -30.
dist = 10.

# to do: define fld_slice
# Animation 1:
nfig += 1
f_title = r''
x_lab = [r'',r'',r'']
y_lab = [r'',r'',r'']
z_lab = [r'',r'',r'']

fig = plt.figure()
ax0 = fig.add_axes([0.05,0.75,0.2,0.2])
ax1 = Axes3D(fig, rect=(0.25,0.5,0.7,0.5), azim=azim, elev=elev, proj_type='ortho')
ax2 = Axes3D(fig, rect=(0.25,0,0.7,0.5), azim=azim, elev=elev, proj_type='ortho')
fld_slice = [ax0,ax1,ax2]
fig_n = 'fld_slice_anim' + str(nfig) + run_ident[0] + '.mp4'

ax0.plot(np.log(en[:,a_i]), pot.V(en[:,phi_i], pot.chi_min(en[:,phi_i]))/pot.V_0(en[:,phi_i],en[:,0]), ls=':', c='k', label=r'$V_{\mathrm{min}}|_{\langle \phi \rangle}/V_0$')
ax0.plot(np.log(en[:,a_i]), en[:,rhoP_i]/pot.V_0(en[:,phi_i],en[:,0]), ls='-.', c='k', label=r'$V/V_0$')
ylim = ax0.get_ylim()
line = ax0.plot([np.log(en[0,a_i]),np.log(en[0,a_i])], ylim ,ls='-', c='g', lw=1)
ax0.set_ylim(ylim)

# to do: set titles and labels
# to do: set figure size
def init_anim1():
	fig.suptitle(f_title)
	for i in range(0,len(fld_slice)):
		fld_slice[i].set_xlabel(x_lab[i])
		fld_slice[i].set_ylabel(y_lab[i])
	for i in range(1,len(fld_slice)):
		fld_slice[i].set_zlabel(z_lab[i])
	#for axis in fld_slice:
	#	axis.set_yticks([]); axis.set_xticks([])
	#fig.set_figheight(4.8*2)
	#fig.set_figwidth(6.4*2)
#	ax0.plot(np.log(en[:,a_i]), pot.V(en[:,phi_i], pot.chi_min(en[:,phi_i]))/pot.V_0(en[:,phi_i],en[:,chi_i]), ls=':', c='k', label=r'$V_{\mathrm{min}}|_{\langle \phi \rangle}/V_0$')
#	ylim = ax0.get_ylim()
	ax1.plot_surface(X,Y,phi[0,:,:,z_i], rcount=nx, cmap='viridis')
	ax2.plot_surface(X,Y,chi[0,:,:,z_i], rcount=nx, cmap='viridis')
	return fld_slice

# to do: update plots
def anim1(t):
	print('Animation 1: frame ', t)
	line[0].set_data([np.log(en[t*sl,a_i]),np.log(en[t*sl,a_i])], ylim)
	ax0.set_ylim(ylim)
	ax1.clear()
	ax2.clear()
	ax1.plot_surface(X,Y,phi[t,:,:,z_i], rcount=nx, cmap='viridis')
	ax2.plot_surface(X,Y,chi[t,:,:,z_i], rcount=nx, cmap='viridis')
	return fld_slice

anim = animation.FuncAnimation(fig, anim1, frames=nl, init_func=init_anim1, interval=300)
#t=2
#ax1.plot_surface(X,Y,phi[t,:,:,z_i], rcount=nx, cmap='viridis')
#ax2.plot_surface(X,Y,chi[t,:,:,z_i], rcount=nx, cmap='viridis')

# Save/show
writer = animation.FFMpegFileWriter(codec='mpeg4')
if SAVE_FIGS:
	#plt.savefig(fig_n)
	anim.save(fig_n, writer=writer)
	

if SHOW_FIGS:
	plt.show()
    
