# zeta_moments_plot.py

# Script to plot zeta moments
# Plot 1: \zeta moments 2, 3, 4 vs ln(a) for all runs
# Plot 2: Difference in \zeta moments from baseline vs ln(a)
# Plot 3: Skew and kurtosis of \zeta vs ln(a)
# Plot 4: Difference in skew and kurtosis of \zeta from baseline vs ln(a)

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from plot_param import *

plt.style.use('./lat_plots.mplstyle')

# Save figs
SAVE_FIGS = False
SHOW_FIGS = True

# Read in data
en_bl, en = load_energy()
zm_bl, zm = load_moments()

# Commpute skew and kurtosis
skew_bl = zm_bl[:,m3]/np.sqrt(zm_bl[:,m2]**3)
skew = zm[:,:,m3]/np.sqrt(zm[:,:,m2]**3)
kurtosis_bl = zm_bl[:,m4]/zm_bl[:,m2]**2
kurtosis = zm[:,:,m4]/zm[:,:,m2]**2

# Make plots
nfig = 0

# Plot 1: \zeta moments 2, 3, 4 vs ln(a) for all runs
nfig += 1
nr = 3; nc =1
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True)
f_title = r'$\langle (\zeta-\langle \zeta \rangle)^n \rangle$'
x_lab = r'$\mathrm{ln}(a)$'
y_lab = [r'$\langle (\zeta-\langle \zeta \rangle)^2 \rangle$', r'$\langle (\zeta-\langle \zeta \rangle)^3 \rangle$', r'$\langle (\zeta-\langle \zeta \rangle)^4 \rangle$']
fig.suptitle(f_title)
ax[-1].set_xlabel(x_lab)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])

ax[0].plot(np.log(en_bl[:,a_i]), zm_bl[:,m2], c='k')
ax[1].plot(np.log(en_bl[:,a_i]), zm_bl[:,m3], c='k')
ax[2].plot(np.log(en_bl[:,a_i]), zm_bl[:,m4], c='k')
for i in range(0,len(zm)):
	ax[0].plot(np.log(en[i,:,a_i]), zm[i,:,m2])
	ax[1].plot(np.log(en[i,:,a_i]), zm[i,:,m3])
	ax[2].plot(np.log(en[i,:,a_i]), zm[i,:,m4])

# Plot 2: Difference in \zeta moments from baseline vs ln(a)
nfig += 1
nr = 3; nc =1
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True)
f_title = r'$\langle (\zeta-\langle \zeta \rangle)^n \rangle - \langle (\zeta_{\Delta V=0} -\langle \zeta_{\Delta V=0} \rangle)^n \rangle$'
x_lab = r'$\mathrm{ln}(a)$'
y_lab = [r'$\Delta\langle (\zeta-\langle \zeta \rangle)^2 \rangle$', r'$\Delta\langle (\zeta-\langle \zeta \rangle)^3 \rangle$', r'$\Delta\langle (\zeta-\langle \zeta \rangle)^4 \rangle$']
fig.suptitle(f_title)
ax[-1].set_xlabel(x_lab)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])

for i in range(0,len(zm)):
	ax[0].plot(np.log(en[i,:,a_i]), zm[i,:,m2] - zm_bl[:,m2])
	ax[1].plot(np.log(en[i,:,a_i]), zm[i,:,m3] - zm_bl[:,m3])
	ax[2].plot(np.log(en[i,:,a_i]), zm[i,:,m4] - zm_bl[:,m4])

# Plot 3: Skew and kurtosis of \zeta vs ln(a)
nfig += 1
nr = 2; nc =1
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True)
f_title = ''
s_title = ['']
x_lab = r'$\mathrm{ln}(a)$'
y_lab = [r'$\frac{\langle (\zeta-\langle \zeta \rangle)^3 \rangle}{\langle (\zeta-\langle \zeta \rangle)^2 \rangle^{\frac{3}{2}}}$', r'$\frac{\langle (\zeta-\langle \zeta \rangle)^4 \rangle}{\langle (\zeta-\langle \zeta \rangle)^2 \rangle^2}$']
fig.suptitle(f_title)
ax[-1].set_xlabel(x_lab)
#for i in range(0,nr):
ax[0].set_ylabel(y_lab[0])
ax[1].set_ylabel(y_lab[1])

ax[0].plot(np.log(en_bl[:,a_i]), skew_bl)
ax[1].plot(np.log(en_bl[:,a_i]), kurtosis_bl-3)
for i in range(0,len(zm)):
	ax[0].plot(np.log(en[i,:,a_i]), skew[i])
	ax[1].plot(np.log(en[i,:,a_i]), kurtosis[i]-3)

# Plot 4: Difference in skew and kurtosis of \zeta from baseline vs ln(a)
nfig += 1
nr = 2; nc =1
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True)
f_title = ''
s_title = ['']
x_lab = r'$\mathrm{ln}(a)$'
y_lab = [r'$\Delta\frac{\langle (\zeta-\langle \zeta \rangle)^3 \rangle}{\langle (\zeta-\langle \zeta \rangle)^2 \rangle^{\frac{3}{2}}}$', r'$\Delta\frac{\langle (\zeta-\langle \zeta \rangle)^4 \rangle}{\langle (\zeta-\langle \zeta \rangle)^2 \rangle^2}$']
fig.suptitle(f_title)
ax[-1].set_xlabel(x_lab)
#for i in range(0,nr):
ax[0].set_ylabel(y_lab[0])
ax[1].set_ylabel(y_lab[1])

for i in range(0,len(zm)):
	ax[0].plot(np.log(en[i,:,a_i]), skew[i] - skew_bl)
	ax[1].plot(np.log(en[i,:,a_i]), kurtosis[i] - kurtosis_bl)
#fig.tight_layout()

if SHOW_FIGS:
	plt.show()
