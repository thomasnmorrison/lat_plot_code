# potential_plt.py

# Script to plot potential

# Plot 1: Plot of m^2_{eff}(\phi) along side surface plot of V(\phi,\chi)
# Plot 2: Contour plot of stable and unstable regions of field space in the
#         \phi and \chi directions.
# Plot 3: Contour plot of stable and unstable regions of field space in the
#         \phi and \chi directions. 
# Plot 4: Contour plot of the potential
# Plot 5: contour plot of V_{,\phi} and V_{,\chi}

# to do:

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
f2_lim = [-1.25*pot.chi_min(phi_p), 1.25*pot.chi_min(phi_p)]
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

# Plot 1: Plot of m^2_{eff}(\phi) along side surface plot of V(\phi,\chi)
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

# Plot 2: Contour plot of stable and unstable regions of field space in the
#         \phi and \chi directions.
# to do: add contour for minimum
# to do: add colour bar
nfig += 1
nr=1; nc=2
fig_n = 'potential_plt' + str(nfig) + '.png'
f_title = r'(Un)stable Regions'
s_title = [r'$V_{,\phi\phi}$',r'$V_{,\chi\chi}$']
x_lab = [r'$\chi$',r'$\chi$']
y_lab = [r'$\phi-\phi_p$',r'$\phi-\phi_p$']
fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
V_f1f1 = pot.ddV(X,Y,1,1)
V_f2f2 = pot.ddV(X,Y,2,2)
cont = 20#np.array([0.])
vlim1 = np.max((np.absolute(np.min(V_f1f1)), np.absolute(np.max(V_f1f1))))
vlim2 = np.max((np.absolute(np.min(V_f2f2)), np.absolute(np.max(V_f2f2))))
vlim = np.absolute(np.min(V_f2f2))
v_11f = ax[0].contourf(Y,X-phi_p,V_f1f1,cont,vmin=-vlim1,vmax=vlim1)
v_22f = ax[1].contourf(Y,X-phi_p,V_f2f2,cont,vmin=-vlim,vmax=vlim)
cont = np.array([0.])
v_11 = ax[0].contour(Y,X-phi_p,V_f1f1,cont,linestyles='--',colors=['w'])
v_22 = ax[1].contour(Y,X-phi_p,V_f2f2,cont,linestyles='--',colors=['w'])
ax[0].clabel(v_11,inline=1,fontsize=10)
ax[1].clabel(v_22,inline=1,fontsize=10)
fig.colorbar(v_11f,ax=ax[0])
fig.colorbar(v_22f,ax=ax[1])
for i in range(0,nc):
    ax[i].set_title(s_title[i])
    ax[i].set_xlabel(x_lab[i])
    ax[i].set_ylabel(y_lab[i])
    #ax[i].legend()
    
if SAVE_FIGS:
    plt.savefig(fig_n)

# Plot 3: Contour plot of stable and unstable regions of field space in the
#         \phi and \chi directions. 
nfig += 1
nr=1; nc=2
fig_n = 'potential_plt' + str(nfig) + '.png'
f_title = r'(Un)stable Regions'
s_title = [r'$\mathrm{sign}(V_{,\phi\phi})\sqrt{|V_{,\phi\phi}|}/m_{\phi\phi}$',r'$\mathrm{sign}(V_{,\chi\chi})\sqrt{|V_{,\chi\chi}|}/m_{\phi\phi}$']
x_lab = [r'$\chi$',r'$\chi$']
y_lab = [r'$\phi-\phi_p$',r'$\phi-\phi_p$']
fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
V_f1f1 = pot.ddV(X,Y,1,1)
V_f2f2 = pot.ddV(X,Y,2,2)
f2_min = pot.chi_min(f1)
cont = 20#np.array([0.])
vlim1 = np.max((np.absolute(np.min(V_f1f1)), np.absolute(np.max(V_f1f1))))
vlim2 = np.max((np.absolute(np.min(V_f2f2)), np.absolute(np.max(V_f2f2))))
vlim = np.absolute(np.min(V_f2f2))
v_11f = ax[0].contourf(Y,X-phi_p,np.sign(V_f1f1)*np.sqrt(np.absolute(V_f1f1)),cont,vmin=-np.sqrt(vlim1),vmax=np.sqrt(vlim1))
v_22f = ax[1].contourf(Y,X-phi_p,np.sign(V_f2f2)*np.sqrt(np.absolute(V_f2f2)),cont,vmin=-np.sqrt(vlim2),vmax=np.sqrt(vlim2))
v_11 = ax[0].contour(Y,X-phi_p,V_f1f1,[0.],linestyles='--',linewidths=0.5,colors=['w'])
v_22 = ax[1].contour(Y,X-phi_p,V_f2f2,[0.],linestyles='--',linewidths=0.5,colors=['w'])
ax[0].clabel(v_11,inline=1,fontsize=10)
ax[1].clabel(v_22,inline=1,fontsize=10)
fig.colorbar(v_11f,ax=ax[0])
fig.colorbar(v_22f,ax=ax[1])
for i in range(0,nc):
	ax[i].plot(f2_min,f1-phi_p, c='k', lw=0.5,label=r'$V_{,\chi}=0$')
	ax[i].plot(-f2_min,f1-phi_p, c='k', lw=0.5)
	ax[i].set_title(s_title[i])
	ax[i].set_xlabel(x_lab[i])
	ax[i].set_ylabel(y_lab[i])
	ax[i].legend()
    
if SAVE_FIGS:
    plt.savefig(fig_n)

# Plot 4: Contour plot of the potential
nfig += 1
nr=1; nc=1
fig_n = 'potential_plt' + str(nfig) + '.png'
f_title = r'Potential Contour Plot'
s_title = [r'']
x_lab = [r'$\chi$',r'$\chi$']
y_lab = [r'$\phi-\phi_p$',r'$\phi-\phi_p$']
fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
V_f1 = pot.V(X,Y)
cont = 20
v_f = ax.contourf(Y,X-phi_p,V_f1,cont)
#v_1 = ax.contour(Y,X-phi_p,V_f1,cont,linestyles='--',linewidths=0.25,colors=['w'])
ax.plot(f2_min,f1-phi_p, c='k', lw=0.5,label=r'$V_{,\chi}=0$')
ax.plot(-f2_min,f1-phi_p, c='k', lw=0.5)

if SAVE_FIGS:
    plt.savefig(fig_n)

# Plot 5: contour plot of V_{,\phi} and V_{,\chi}
nfig += 1
nr=1; nc=2
fig_n = 'potential_plt' + str(nfig) + '.png'
f_title = r'Potential Contour Plot'
s_title = [r'$V_{,\phi}/m^2_{\phi\phi}$',r'$V_{,\chi}/m^2_{\phi\phi}$']
x_lab = [r'$\chi$',r'$\chi$']
y_lab = [r'$\phi-\phi_p$',r'$\phi-\phi_p$']
fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
V_f1 = pot.dV(X,Y,1)
V_f2 = pot.dV(X,Y,2)
cont = 20
v_1f = ax[0].contourf(Y,X-phi_p,V_f1,cont)
v_2f = ax[1].contourf(Y,X-phi_p,V_f2,cont)
#ax[0].contour(Y,X-phi_p,V_f1,cont,linestyles='-',linewidths=0.25,colors=['w'])
#ax[1].contour(Y,X-phi_p,V_f2,cont,linestyles='-',linewidths=0.25,colors=['w'])
v_1 = ax[0].contour(Y,X-phi_p,V_f1,[0.],linestyles='--',linewidths=0.5,colors=['w'])
v_2 = ax[1].contour(Y,X-phi_p,V_f2,[0.],linestyles='--',linewidths=0.5,colors=['w'])
ax[0].clabel(v_1,inline=1,fontsize=10)
ax[1].clabel(v_2,inline=1,fontsize=10)
fig.colorbar(v_1f,ax=ax[0])
fig.colorbar(v_2f,ax=ax[1])
for i in range(0,nc):
	ax[i].plot(f2_min,f1-phi_p, c='k', lw=0.5,label=r'$V_{,\chi}=0$')
	ax[i].plot(-f2_min,f1-phi_p, c='k', lw=0.5)
	ax[i].set_title(s_title[i])
	ax[i].set_xlabel(x_lab[i])
	ax[i].set_ylabel(y_lab[i])
	ax[i].legend()

if SAVE_FIGS:
    plt.savefig(fig_n)


if SHOW_FIGS:
    plt.show()
