# ener_plot.py

# Script for lattice average energy plots
# Plot 1: ln(a) vs 1/(aH) for baseline and 
# Plot 2: ln(a) vs fractional KE, PE, GE
# Plot 3: ln(a) vs -dln(H)/dln(a)
# Plot 4: ln(a) vs \bar{\phi}, \bar{\chi}, \dot{\bar{\phi}}, \dot{\bar{\chi}}
# Plot 5: ln(a) vs conservation of energy

# to do: plot lines at \Delta V
# to do: plot \Delta V\V_0

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from plot_param import *
import potential as pot

# Set style
plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = True
SHOW_FIGS = True

# Read in data
en_bl, en = load_energy()

# Find index of start/end of \Delta V
dv_i = np.zeros(2, dtype=np.int64)
dv_i[0] = np.argmin(np.absolute(en[0,:,phi_i]-(phi_p+phi_w)))
dv_i[1] = np.argmin(np.absolute(en[0,:,phi_i]-(phi_p-phi_w)))

# Set potential parameters
pot.init_param(phi_p, phi_w, m2_p, lambda_chi)

# Make plots
nfig = 0

# Plot 1: ln(a) vs 1/(aH)
# to do: add units
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Horizon'
s_title = [r'Horizon']
x_lab = [r'$\alpha$']#[r'$\mathrm{ln}(a)$']
y_lab = [r'$(aH)^{-1}$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0]); ax.set_ylabel(y_lab[0])
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xmargin(m = 0.)

ax.plot(np.log(en_bl[:,a_i]), 1./(en_bl[:,a_i]*en_bl[:,hub_i]), ls='--', c='k')
for i in range(0,len(en)):
	ax.plot(np.log(en[i,:,a_i]), 1./(en[i,:,a_i]*en[i,:,hub_i]), ls='-', c=c_ar[i%len(c_ar)])
#plt.tight_layout()
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 2: ln(a) vs fractional KE, PE, GE
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Energy Fractions'
s_title = [r'Energy Fractions']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$\rho_{\mathrm{part}}/\rho_{\mathrm{total}}$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0]); ax.set_ylabel(y_lab[0])
ax.set_xmargin(m = 0.)
ax.semilogy()

ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoK_i]/en_bl[:,rho_i], ls='--', c='k', label=r'Kinetic')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoG_i]/en_bl[:,rho_i], ls='--', c='r', label=r'Gradient')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoP_i]/en_bl[:,rho_i], ls='--', c='b', label=r'Potential')
for i in range(0,len(en)):
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoK_i]/en[i,:,rho_i], ls='-', c='k', label=r'Kinetic')
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoG_i]/en[i,:,rho_i], ls='-', c='r', label=r'Gradient')
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoP_i]/en[i,:,rho_i], ls='-', c='b', label=r'Potential')
ylim = ax.get_ylim()
ax.plot([np.log(en[0,dv_i[0],a_i]),np.log(en[0,dv_i[0],a_i])], ylim ,ls='-', c='g', lw=0.5)
ax.plot([np.log(en[0,dv_i[1],a_i]),np.log(en[0,dv_i[1],a_i])], ylim ,ls='-', c='g', lw=0.5, label=r'$\Delta V$ on/off')
ax.set_ylim(ylim)
ax.legend()

if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 2a: ln(a) vs fractional KE, PE, GE
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Energy Fractions'
s_title = [r'Energy Fractions']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$\rho_{\mathrm{part}}/\rho_{\mathrm{total}}$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0]); ax.set_ylabel(y_lab[0])
ax.set_xmargin(m = 0.)
ax.semilogy()

ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoK_i]/en_bl[:,rho_i], ls='--', c='k', label=r'Kinetic')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoG_i]/en_bl[:,rho_i], ls='--', c='r', label=r'Gradient')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoP_i]/en_bl[:,rho_i], ls='--', c='b', label=r'Potential')
for i in range(0,len(en)):
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoK_i]/en[i,:,rho_i], ls='-', c='k', label=r'Kinetic')
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoG_i]/en[i,:,rho_i], ls='-', c='r', label=r'Gradient')
	ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoP_i]/en[i,:,rho_i], ls='-', c='b', label=r'Potential')
        ax.plot(np.log(en[i,:,a_i]), en[i,:,rhoP_i]/pot.V_0(en[i,:,phi_i],en[i,:,chi_i]), ls='-.', c='g', label=r'$V/V_0$')
        ax.plot(np.log(en[i,:,a_i]), pot.V(en[i,:,phi_i], pot.chi_min(en[i,:,phi_i]))/pot.V_0(en[i,:,phi_i],en[i,:,chi_i]), ls='..', c='g', label=r'$V_{\mathrm{min}}|_{\langle \phi \rangle}/V_0$')
ylim = ax.get_ylim()
ax.plot([np.log(en[0,dv_i[0],a_i]),np.log(en[0,dv_i[0],a_i])], ylim ,ls='-', c='g', lw=0.5)
ax.plot([np.log(en[0,dv_i[1],a_i]),np.log(en[0,dv_i[1],a_i])], ylim ,ls='-', c='g', lw=0.5, label=r'$\Delta V$ on/off')
ax.set_ylim(ylim)
ax.legend()

if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 3: (\rho+P)/\rho vs ln(a)
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Equation of State'
s_title = [r'W']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$P/\rho$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0]); ax.set_ylabel(y_lab[0])

w = (en_bl[:,rhoK_i] - en_bl[:,rhoG_i]/3. - en_bl[:,rhoP_i])/en_bl[:,rho_i]
ax.plot(np.log(en_bl[:,a_i]), w, ls='--', c='k')
for i in range(0,len(en)):
	w = (en[i,:,rhoK_i] - en[i,:,rhoG_i]/3. - en[i,:,rhoP_i])/en[i,:,rho_i]
	ax.plot(np.log(en[i,:,a_i]), w, ls='-', c=c_ar[i%len(c_ar)])
ax.legend()
fig.tight_layout()

if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 4: ln(a) vs -dln(H)/dln(a)
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Slow-roll'
s_title = [r'$\epsilon$']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$-\frac{\mathrm{dln}(H)}{\mathrm{dln}(a)}$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0]); ax.set_ylabel(y_lab[0])
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xmargin(m = 0.)
#ax.semilogy()

epsilon = -np.diff(np.log(en_bl[:,hub_i]))/np.diff(np.log(en_bl[:,a_i]))
ax.plot((np.log(en_bl[:-1,a_i])+np.log(en_bl[1:,a_i]))/2., epsilon, ls='--',c='k')
for i in range(0,len(en)):
	epsilon = -np.diff(np.log(en[i,:,hub_i]))/np.diff(np.log(en[i,:,a_i]))
	ax.plot((np.log(en[i,:-1,a_i])+np.log(en[i,1:,a_i]))/2., epsilon, ls='-',c=c_ar[i%len(c_ar)])
ylim = ax.get_ylim()
ax.plot([np.log(en[0,dv_i[0],a_i]),np.log(en[0,dv_i[0],a_i])], ylim ,ls='-', c='g', lw=0.5)
ax.plot([np.log(en[0,dv_i[1],a_i]),np.log(en[0,dv_i[1],a_i])], ylim ,ls='-', c='g', lw=0.5, label=r'$\Delta V$ on/off')
ax.set_ylim(ylim)

if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 5: ln(a) vs \bar{\phi}, \bar{\chi}, \dot{\bar{\phi}}, \dot{\bar{\chi}}
nfig += 1
nr = 2; nc = 2
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Mean Fields'
s_title = [r'$\bar{\phi}$', r'$\bar{\chi}$', r'$\dot{\bar{\phi}}$', r'$\dot{\bar{\chi}}$']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$\langle\phi\rangle M_{Pl}$', r'$\langle\chi\rangle M_{Pl}$', 
r'$\langle\dot{\phi}\rangle m_\phi M_{Pl}$', r'$\langle\dot{\chi}\rangle m_\phi M_{Pl}$']
fig.suptitle(f_title)
for j in range(0,nc):
	ax[-1,j].set_xlabel(x_lab[0])
	for i in range(0,nr):
		#ax[i,j].set_title(s_title[2*i+j])	
		ax[i,j].set_ylabel(y_lab[2*i+j])
#		ax[i,j].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
for i in range(0,nr):
	ax[0,i].plot(np.log(en_bl[:,a_i]), en_bl[:,phi_i+2*i], ls='--', c='k')
	ax[1,i].plot(np.log(en_bl[:,a_i]), en_bl[:,dphi_i+2*i]/en_bl[:,a_i]**3, ls='--', c='k')
	for k in range(0,len(en)):
		ax[0,i].plot(np.log(en[k,:,a_i]), en[k,:,phi_i+2*i], ls='-', c=c_ar[k%len(c_ar)])
		ax[1,i].plot(np.log(en[k,:,a_i]), en[k,:,dphi_i+2*i]/en_bl[:,a_i]**3, ls='-', c=c_ar[k%len(c_ar)])
fig.tight_layout()

if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 6: ln(a) vs conservation of energy
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
fig_n = 'ener_plt' + str(nfig) + run_ident[0] + '.png'
f_title = 'Energy Consevervation'
s_title = ['']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$a^4(\rho_\mathrm{Fields}-\rho_\mathrm{Hubble})/\rho_\mathrm{Fields}$']
fig.suptitle(f_title)
ax.set_xlabel(x_lab[0])
ax.set_ylabel(y_lab[0])
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xmargin(m = 0.)
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,a_i]**4*en_bl[:,rhoCons_i], ls='--', c='k', label=r'')
#ax.plot(en_bl[:,phi_i], en_bl[:,a_i]**4*en_bl[:,rhoCons_i], ls='--', c='k', label=r'')
for i in range(0,len(en)):
	ax.plot(np.log(en[i,:,a_i]), en[i,:,a_i]**4*en[i,:,rhoCons_i], ls='-', c=c_ar[i%len(c_ar)], label=r'')
	#ax.plot(en[i,:,phi_i], en[i,:,a_i]**4*en[i,:,rhoCons_i], ls='-', c=c_ar[i%len(c_ar)], label=r'')

if SAVE_FIGS:
	plt.savefig(fig_n)

plt.show()
