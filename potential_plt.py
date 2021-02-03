# potential_plt.py

# Script to plot potential

# to do: invert x axis on Plot 1a

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

# Set potential parameters
pot.init_param(phi_p, phi_w, m2_p, lambda_chi_in=lambda_chi, POTOPT_in=POTOPT)

# Set field limits
n_pts = 2**8
f1_lim = [phi_p+1.25*phi_w, phi_p-1.25*phi_w]
f2_lim = [-2.*pot.chi_min(phi_p), 2.*pot.chi_min(phi_p)]
f1 = np.linspace(f1_lim[0],f1_lim[1],n_pts)
f2 = np.linspace(f2_lim[0],f2_lim[1],n_pts)
X, Y = np.meshgrid(f1, f2)

v = pot.V(X,Y)#pot.V_int(X,Y)+pot.V_0_chi(X,Y)#
clip_lim = pot.V(X,0,baseline=True) + pot.V_0_chi(0,1.*pot.chi_min(phi_p))
v_clip = np.where(v<=clip_lim, v, np.nan)

# Plot
nfig = 0

# Point of view
elev = 45.
azim = 135.
dist = 10.

# Plot 1:
nfig += 1
fig_n = 'potential_plt' + str(nfig) + '_' + str(POTOPT) + '_' + '.png'
x_lab = [r'$\phi-\phi_p$',r'$\phi-\phi_p$']
y_lab = [r'',r'$\chi$']
z_lab = [r'',r'$V(\phi,\chi)$']
l_lab = [r'$m^2_{\mathrm{eff}(\phi)}/m_{\phi\phi}^2$',r'$V(\phi,\chi)$']

fig = plt.figure()
ax0 = fig.add_axes([0.1,0.55,0.8,0.4])
ax1 = Axes3D(fig, rect=(0.1,0.05,0.8,0.4), azim=azim, elev=elev, proj_type='ortho')
ax = [ax0,ax1]
for i in range(0,len(ax)):
	ax[i].set_xlabel(x_lab[i])
	ax[i].set_ylabel(y_lab[i])
ax1.set_zlabel(z_lab[1])
ax0.plot(f1-phi_p, pot.m2eff_chi(f1), c='b', ls='-', lw=1, label=l_lab[0])
ax0.invert_xaxis()
#ax1.plot_surface(X,Y,pot.V_int(X,Y)+pot.V_0_chi(X,Y), rcount=n_pts, cmap='coolwarm')
ax1.plot_surface(X-phi_p,Y,v_clip, rcount=n_pts, cmap='plasma', vmin=np.nanmin(v_clip), vmax=np.nanmax(v_clip))
for axis in ax:
    axis.legend()
    
if SAVE_FIGS:
    plt.savefig(fig_n)

if SHOW_FIGS:
    plt.show()
