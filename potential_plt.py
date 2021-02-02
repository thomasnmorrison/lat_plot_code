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
n_pts = 2**7
f1_lim = [phi_p+2.*phi_w, phi_p-2.*phi_w]
f2_lim = [-2.*pot.chi_min(phi_p), 2.*pot.chi_min(phi_p)]
f1 = np.linspace(f1_lim[0],f1_lim[1],n_pts)
f2 = np.linspace(f2_lim[0],f2_lim[1],n_pts)
X, Y = np.meshgrid(f1, f2)

# Plot
nfig = 0

# Point of view
elev = 30.
azim = 150.
dist = 10.

# Plot 1:
nfig += 1
fig_n = 'potential_plt' + str(nfig) + '.png'
x_lab = [r'',r'']
y_lab = [r'',r'']
z_lab = [r'',r'']
l_lab = [r'$m^2_{\mathrm{eff}(\phi)}$',r'$V(\phi,\chi)$']

fig = plt.figure()
ax0 = fig.add_axes([0.05,0.5,0.9,0.5])
ax1 = Axes3D(fig, rect=(0.05,0.0,0.9,0.5), azim=azim, elev=elev, proj_type='ortho')
ax = [ax0,ax1]
ax0.plot(f1, pot.m2eff_chi(f1), c='b', ls='-', lw=1, label=l_lab[0])
ax1.plot_surface(X,Y,pot.V(X,Y), rcount=n_pts, cmap='viridis')
for axis in ax:
    axis.legend()
    
if SAVE_FIGS:
    plt.savefig(fig_n)

if SHOW_FIGS:
    plt.show()
