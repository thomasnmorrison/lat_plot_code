# ener_plot.py

# Script for lattice average energy plots
# Plot 1: ln(a) vs 1/(aH)
# Plot 2: ln(a) vs fractional KE, PE, GE
# Plot 3: ln(a) vs -dln(H)/dln(a)
# Plot 4: ln(a) vs \bar{\phi}, \bar{\chi}, \dot{\bar{\phi}}, \dot{\bar{\chi}}
# Plot 5:

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# File names and paths
path_n = '../lattice-dev-master/Pseudospec/openmp_dev/'
en_bl_f = 'energy_spec_TESTING_.out' # Baseline run lattice averaged quantities
en_f = [] # lattice averaged quatities

# Configuration parameters
#BASE_LINE:    Plot the baseline run
#ALL_RUN:      Plot all runs
#DELTA:        Plot all runs differenced from the baseline
#SAVE_FIG
#SCALE_FIG
#FIG_SIZE
#WATER_MARK
BASE_LINE = [True, True, True]
DELTA = [False, False, False]
ALL_RUN = [False, False, False]
SAVE_FIG = [False, False, False]
SCALE_FIG = [False, False, False]
FIG_SIZE = (3.50, 2.16)
WATER_MARK = [False, False, False]

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14

# Plot output file names


# Read in data
en_bl = np.loadtxt(path_n+en_bl_f)

en = np.zeros((np.shape(en_f) + np.shape(en_bl)))
for i in range(0,len(en_f)):
	en[i] = np.loadtxt(path_n+en_f[i])

print('np.shape(en_bl) = ', np.shape(en_bl))
# Indexing constants
a_i = 1; rho_i = 2; rhoK_i = 3; rhoP_i = 4; rhoG_i = 5; hub_i = 8; phi_i = 9
# Calc H, \dot{\phi}
en_bl[:,hub_i] = np.sqrt(-en_bl[:,hub_i]/3.)

# Make plots
nfig = 0

# Plot 1: ln(a) vs 1/(aH)
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'Horizon'
s_title = [r'Horizon']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$(aH)^{-1}$']
ax.set_title(s_title[0], fontsize = title_fs)
ax.set_xlabel(x_lab[0], fontsize = x_lab_fs); ax.set_ylabel(y_lab[0], fontsize = y_lab_fs)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xmargin(m = 0.)

ax.plot(np.log(en_bl[:,a_i]), 1./(en_bl[:,a_i]*en_bl[:,hub_i]), ls='--', c='k')
#plt.tight_layout()

if (SCALE_FIG[nfig-1] == True):
	fig.set_size_inches(FIG_SIZE)

# Plot 2: ln(a) vs fractional KE, PE, GE
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'Energy Fractions'
s_title = [r'Energy Fractions']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$\rho_{\mathrm{part}}/\rho_{\mathrm{total}}$']
ax.set_title(s_title[0], fontsize = title_fs)
ax.set_xlabel(x_lab[0], fontsize = x_lab_fs); ax.set_ylabel(y_lab[0], fontsize = y_lab_fs)
ax.set_xmargin(m = 0.)
ax.semilogy()

ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoK_i]/en_bl[:,rho_i], ls='--', c='k', label=r'Kinetic')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoG_i]/en_bl[:,rho_i], ls='--', c='r', label=r'Gradient')
ax.plot(np.log(en_bl[:,a_i]), en_bl[:,rhoP_i]/en_bl[:,rho_i], ls='--', c='b', label=r'Potential')
ax.legend()

# Plot 3: ln(a) vs -dln(H)/dln(a)
nfig += 1
fig, ax = plt.subplots(nrows=1 , ncols=1 , sharex=False)
f_title = r'$\epsilon$'
s_title = [r'$\epsilon$']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$-\frac{\mathrm{dln}(H)}{\mathrm{dln}(a)}$']
ax.set_title(s_title[0], fontsize = title_fs)
ax.set_xlabel(x_lab[0], fontsize = x_lab_fs); ax.set_ylabel(y_lab[0], fontsize = y_lab_fs)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xmargin(m = 0.)
ax.semilogy()

epsilon = -np.diff(np.log(en_bl[:,hub_i]))/np.diff(np.log(en_bl[:,a_i]))
ax.plot((np.log(en_bl[:-1,a_i])+np.log(en_bl[1:,a_i]))/2., epsilon)

# Plot 4: ln(a) vs \bar{\phi}, \bar{\chi}, \dot{\bar{\phi}}, \dot{\bar{\chi}}
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = r'Mean Fields'
s_title = [r'$\bar{\phi}$', r'$\bar{\chi}$', r'$\dot{\bar{\phi}}$', r'$\dot{\bar{\chi}}$']
x_lab = [r'$\mathrm{ln}(a)$']
y_lab = [r'$M_{Pl}$', r'$M_{Pl}$', r'$m_\phi M_{Pl}$', r'$m_\phi M_{Pl}$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].set_title(s_title[2*i+j], fontsize = stitle_fs)
		ax[i,j].set_xlabel(x_lab[0], fontsize = x_lab_fs)
		ax[i,j].set_ylabel(y_lab[2*i+j], fontsize = y_lab_fs)
		ax[i,j].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
		ax[i,j].set_xmargin(m = 0.)
	ax[0,i].plot(np.log(en_bl[:,a_i]), en_bl[:,phi_i+2*i])
	ax[1,i].plot(np.log(en_bl[:,a_i]), en_bl[:,dphi_i+2*i]/en_bl[:,a_i]**3)

# Plot 5:
nfig += 1
fig, ax = plt.subplots(nrows=2 , ncols=2 , sharex=False)
f_title = ''
s_title = ['']






plt.show()
