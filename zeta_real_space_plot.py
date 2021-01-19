# zeta_real_space_plot.py

# Script for plotting realspace slices of \zeta at sampled times.

# Plot 1: Colormap of \Delta\zeta and \Delta\dot{\zeta}/H and 
#         \Delta\dot{\zeta}/\sqrt{\langle \dot{\zeta}_{\Delta V=0}^2 \rangle} on a 2d slice at specified time slices
# Plot 2: Colormap of \Delta\dot{\zeta}_{part}/H on a 2d slice at specified time slices
# Plot 3: Colormap of \Delta\dot{\zeta}_{part}/\sqrt{\langle \dot{\zeta}_{\Delta V=0}^2 \rangle}
#         on a 2d slice at specified time slices

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg
from plot_param import *

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read-in data
en_bl, en = load_energy()
zeta = load_lat(zeta_f[0], path[0])
zeta_bl = load_lat(zeta_bl_f, path_bl)
#dz_lapp_bl = load_lat()
#dz_gdgp_bl
#dz_lapc_bl
#dz_gdgc_bl
#dz_lapp
#dz_gdgp
#dz_lapc
#dz_gdgc
dz_lapp_bl, dz_gdgp_bl, dz_lapc_bl, dz_gdgc_bl, dz_lapp, dz_gdgp, dz_lapc, dz_gdgc = load_dzeta_part()

dzeta_p = [[dz_lapp_bl, dz_gdgp_bl, dz_lapc_bl, dz_gdgc_bl],[dz_lapp[0], dz_gdgp[0], dz_lapc[0], dz_gdgc[0]]]
dzeta_p = np.array(dzeta_p)
dz_lf1_bl_i = [0,0]; dz_gf1_bl_i = [0,1]; dz_lf2_bl_i = [0,2]; dz_gf2_bl_i = [0,3]
dz_lf1_i = [1,0]; dz_gf1_i = [1,1]; dz_lf2_i = [1,2]; dz_gf2_i = [1,3]
dz_i = np.array([dz_lf1_bl_i, dz_gf1_bl_i, dz_lf2_bl_i, dz_gf2_bl_i, dz_lf1_i, dz_gf1_i, dz_lf2_i, dz_gf2_i])

dzeta_p = np.reshape(dzeta_p, (2,2*nfld,nl,nx,ny,nz))
zeta_bl = np.reshape(zeta_bl, (nl,nx,ny,nz))
zeta = np.reshape(zeta, (-1,nl,nx,ny,nz))

Ddzeta = 

print('Data read-in complete')

# Choose slices
t_i = np.zeros(5,np.int64)  # time indicies of slices to show
t_i[1] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p + phi_w)))
t_i[2] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p)))
t_i[3] = np.argmin(np.absolute(fld0[f1_i[0],f1_i[1]] - (phi_p - phi_w)))
t_i[-1] = -1
t_i = [32,40,48,56,64]
z_i = 32  # z index of spatial slice

# Set histogram limits
vz = np.zeros(2)
vdz = np.zeros(2)
vdz_n = np.zeros(2)
vdzp = np.zeros((2*nfld,2))
vdzp_n = np.zeros((2*nfld,2))
vz_m = np.min((zeta-zeta_bl).T)
vz_p = np.max((zeta-zeta_bl).T)
vdz_m = np.min()
vdz_p = np.max()
vdz_n_m = np.min()
vdz_n_p = np.max()
vz = np.array([vz_m,vz_p])
vdz = np.array([vdz_m,vdz_p])
vdz_n = np.array([vdz_n_m,vdz_n_p])
for i in range(0,2*nfld):
	vdzp_m = np.min((dzeta_p[dz_i[2*nfld+j,0],dz_i[2*nfld+j,1]].T - dzeta_p[dz_i[j,0],dz_i[j,1]].T)/(en[0,::sl,a_i]*en[0,::sl,hub_i]))
	vdzp_p = np.max((dzeta_p[dz_i[2*nfld+j,0],dz_i[2*nfld+j,1]].T - dzeta_p[dz_i[j,0],dz_i[j,1]].T)/(en[0,::sl,a_i]*en[0,::sl,hub_i]))
	vdzp_n_m = np.min((dzeta_p[dz_i[2*nfld+i,0],dz_i[2*nfld+i,1]].T - dzeta_p[dz_i[i,0],dz_i[i,1]].T)/dzeta_std)
	vdzp_n_p = np.max((dzeta_p[dz_i[2*nfld+i,0],dz_i[2*nfld+i,1]].T - dzeta_p[dz_i[i,0],dz_i[i,1]].T)/dzeta_std)
	vdzp[i] = [-np.max([np.absolute(vdzp_m),np.absolute(vdzp_p)]),np.max([np.absolute(vdzp_m),np.absolute(vdzp_p)])]
	vdzp_n[i] = [-np.max([np.absolute(vdzp_n_m),np.absolute(vdzp_n_p)]),np.max([np.absolute(vdzp_n_m),np.absolute(vdzp_n_p)])]


# Make plots
nfig = 0









if SHOW_FIGS:
	plt.show()
