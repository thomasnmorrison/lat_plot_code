# fld_traj_zeta_condition.py

# Script to plot the field, momentum, and \zeta trajectories conditioned on \Delta\zeta_{final}.

# Plot 1: Plot of \phi and \dot{\phi}.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
# Plot 2: Plot of \chi and \dot{\chi}.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
# Plot 3: Plot of \Delta\zeta.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
# Plot 4: Plot of \Delta\phi and \Delta\dot{\phi}.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
# Plot 5: Plot of \Delta\chi and \Delta\dot{\chi}.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.

# to do: make only one cut with all trajectories
# to do: reindex axes for one cut

# Import packages
import numpy as np
import matplotlib.pyplot as plt
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

# Apply cuts to include all trajectories in just one cut
# cut_i: index of slice, which is all lattice sites here
# cut: tuple of (ind on lattice), cast to an np array
#phi_cut = phi_p  # mean \phi determining the time at which the cut is applied
#cut_i = np.argmin(np.abs(en[0,::sl,phi_i] - phi_cut))
#cut = (np.argwhere(np.where((chi[0,cut_i,:]-chi_bl[cut_i,:])>=0.,1,0)), np.argwhere(np.where((chi[0,cut_i,:]-chi_bl[cut_i,:])<0.,1,0)))
cut = (np.argwhere(np.ones(nlat)),)
cut = np.array(cut)
cut = cut[:,:,0]
print('np.shape(cut) = ', np.shape(cut))
for i, c in enumerate(cut):
	cut[i] = c.flatten()
print('np.shape(cut) = ', np.shape(cut))

# Sort cut data by \Delta\zeta_{end}
# Dz_cut_sort: tuple of (sorted \Delta\zeta_{end} of + \chi traj, sorted \Delta\zeta_{end} of - \chi traj)
# Dz_cut_sort_i: tuple of (ind to sort \Delta\zeta_{end} of + \chi traj, ind to sort \Delta\zeta_{end} of - \chi traj)
# Dz_cut_sort_i: formatted []
Dz_cut_sort = ()
Dz_cut_sort_i = ()
print('Sorting')
for c in cut:
	Dz_cut_sort = Dz_cut_sort + (np.sort((zeta[0]-zeta_bl)[-1,c],axis=-1),)
	Dz_cut_sort_i = Dz_cut_sort_i + (np.argsort((zeta[0]-zeta_bl)[-1,c],axis=-1),)
print('Sorting complete')

# Define trajectory bins, bin by \Delta\zeta_{end}
# traj_cut_bins_i: tuple of (ind of cut for bin edges of \Delta\zeta_{end} for + \chi traj, ind of cut for bin edges of \Delta\zeta_{end} for + \chi traj), 
# traj_bin_label: list of bi labels for plot legends
sig_Dz = np.std((zeta-zeta_bl)[0,-1,:], ddof=-1)   # \Delta\zeta_{end} std
traj_bins = np.array([-5,5])*sig_Dz             # trajector bin edges in terms of std of \Delta\zeta
traj_bin_label = ['']*(len(traj_bins)+1)
traj_bin_label[0] = r'$<{:03.2f}\sigma$'.format(traj_bins[0]/sig_Dz)
#traj_bins_i = np.zeros(len(traj_bins)+2, dtype=np.int64)
#traj_bins_i[-1] = nlat-1                          # outside bin edge
#traj_bins_i[1:-1] = np.searchsorted(Dz_sort, traj_bins) # array of indices in Dz_sort_i that correspond to the bins in traj_bins
for i in range(1,len(traj_bins)):
	traj_bin_label[i] = r'$({:03.2f},{:03.2f})\sigma$'.format(traj_bins[i-1]/sig_Dz, traj_bins[i]/sig_Dz)
traj_bin_label[-1] = r'$>{0}\sigma$'.format(traj_bins[-1]/sig_Dz)
traj_cut_bins_i = ()
for i, c in enumerate(cut):
	cut_bins = np.zeros(len(traj_bins)+2, dtype=np.int64)
	cut_bins[-1] = len(c)-1
	cut_bins[1:-1] = np.searchsorted(Dz_cut_sort[i], traj_bins)
	traj_cut_bins_i = traj_cut_bins_i + (cut_bins,)
traj_cut_bin_full = np.array([[False]*(len(traj_bins)+1)]*len(cut))

# Define bin contours on cut data
# fld_cut_cz: Formatted [[fld], cut, bin, cont, t]
# Delta_fld_cut_cz: Formatted [[fld], cut, bin, cont, t]
# Delta_zeta_cut_cz: Formatted [cut, bin, cont, t]
cont = np.array([0., 0.05, 0.25, 0.75, 0.95, 1.])  # contour levels
fld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl)) 
dfld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_fld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_dfld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_zeta_cut_cz = np.zeros((len(cut), len(traj_bins)+1, len(cont), nl))

# Find cut bin contours
for i, c in enumerate(cut):
	for j in range(1,len(traj_bins)+2):
		n_traj = traj_cut_bins_i[i][j] - traj_cut_bins_i[i][j-1]   # Find number of trajectories in bin for this cut	
		ind = c[Dz_cut_sort_i[i][traj_cut_bins_i[i][j-1]:traj_cut_bins_i[i][j]]]  # Find indices of trajectories in this bin for this cut
		if (len(ind)>0):
			traj_cut_bin_full[i,j-1] = True  # if this bin covers at least one trajectory set to True
		for f_i in fld_i:
			# Sort fields in this cut
			fld_sort = np.sort(fld[f_i[0],f_i[1]][:,ind], axis=-1).T    # Formatted [traj, time]
			dfld_sort = np.sort(dfld[f_i[0],f_i[1]][:,ind], axis=-1).T  # Formatted [traj, time]
			for k, d in enumerate(cont):
				if (n_traj > 0):
					cont_ind = int(d*(n_traj - 1))                # index of contour
					fld_cut_cz[f_i[0],f_i[1]][i,j-1,k,:] = fld_sort[cont_ind,:]
					dfld_cut_cz[f_i[0],f_i[1]][i,j-1,k,:] = dfld_sort[cont_ind,:]
		for l in range(0,nfld):
			# Sort \Delta fields in this cut
			fld_sort = np.sort(fld[1,l][:,ind] - fld[0,l][:,ind], axis=-1).T    # Formatted [traj, time]
			dfld_sort = np.sort(dfld[1,l][:,ind] - dfld[0,l][:,ind], axis=-1).T  # Formatted [traj, time]
			for k, d in enumerate(cont):
				if (n_traj > 0):
					cont_ind = int(d*(n_traj - 1))                # index of contour
					Delta_fld_cut_cz[1,l][i,j-1,k,:] = fld_sort[cont_ind,:]
					Delta_dfld_cut_cz[1,l][i,j-1,k,:] = dfld_sort[cont_ind,:]
		# Sort \Delta\zeta in this cut
		zeta_sort = np.sort((zeta[0]-zeta_bl)[:,ind], axis=-1).T    # Formatted [traj, time]
		for k, d in enumerate(cont):
			if (n_traj > 0):
				cont_ind = int(d*(n_traj - 1))                # index of contour
				Delta_zeta_cut_cz[i,j-1,k,:] = zeta_sort[cont_ind,:]

# Make plots
nfig = 0

bin_select = [0,1,2]

# Plot 1: Plot of \phi and \Pi_\phi cut by \chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
fig_n = 'fld_traj_zc_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'$\Delta\chi(\phi=\phi_p)>0$',r'$\Delta\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\phi-\langle\phi\rangle$', r'$\dot{\phi}-\langle\dot{\phi}\rangle$',
r'$\chi-\langle\chi\rangle$', r'$\dot{\chi}-\langle\dot{\chi}\rangle$']
fig.suptitle(f_title)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[i].set_xlabel(x_lab[-1])
	#ax[i].set_title(s_title[i])

print('traj_cut_bin_full = ', traj_cut_bin_full)
#bin_select = [0,2]
i = 0
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[0].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[1].fill_between(np.log(en[0,::sl,a_i]), ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
#bin_select = [0,1]
ns = 8  # number of sampled trajectories
i = 0
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			if (traj_cut_bins_i[l][j+1] < len(cut[l])): # replace nlat with number of lattice sites in this cut
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[0].plot(np.log(en[0,::sl,a_i]), ((fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]) - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), ((dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax.flatten():
	axis.legend(loc=2)
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 2: Plot of \chi and \dot{\phi} cut by \Delta\chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
fig_n = 'fld_traj_zc_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'$\Delta\chi(\phi=\phi_p)>0$',r'$\Delta\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\chi-\langle\chi\rangle$', r'$\dot{\chi}-\langle\dot{\chi}\rangle$']
fig.suptitle(f_title)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1].set_xlabel(x_lab[-1])
	#ax[0,i].set_title(s_title[i])

print('traj_cut_bin_full = ', traj_cut_bin_full)
#bin_select = [0,2]
i = 1
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[0].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[1].fill_between(np.log(en[0,::sl,a_i]), ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
#bin_select = [0,1]
ns = 8  # number of sampled trajectories
i = 1
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			if (traj_cut_bins_i[l][j+1] < len(cut[l])):
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[0].plot(np.log(en[0,::sl,a_i]), ((fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]) - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), ((dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax.flatten():
	axis.legend(loc=2)
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 3: Plot of \Delta\zeta cut by \Delta\chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 1; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=True)
fig_n = 'fld_traj_zc_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'$\Delta\chi(\phi=\phi_p)>0$',r'$\Delta\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\zeta$']
fig.suptitle(f_title)
ax.set_ylabel(y_lab[0])
for i in range(0,nc):
	ax.set_xlabel(x_lab[-1])
	#ax[i].set_title(s_title[i])

# Plot contours
#bin_select = [0,1]  # select bins to plot contours
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax.fill_between(np.log(en[0,::sl,a_i]), (Delta_zeta_cut_cz[l,j,k,:]).T, (Delta_zeta_cut_cz[l,j,-1-k,:]).T, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])

# Plot trajectories
#bin_select = [0,1]  # select bins to plot sampled trajectories
ns = 10  # number of sampled trajectories
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			if (traj_cut_bins_i[l][j+1] < len(cut[l])):
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax.plot(np.log(en[0,::sl,a_i]), (zeta[0]-zeta_bl)[:,ind_u], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax.plot(np.log(en[0,::sl,a_i]), (zeta[0]-zeta_bl)[:,ind_l], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			# Plot sampled trajectories in bin
			ax.plot(np.log(en[0,::sl,a_i]), ((zeta[0]-zeta_bl)[:,ind_s]), lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

ax.legend(loc=2)
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 4: Plot of \Delta\phi and \Delta\dot{\phi} cut by \Delta\chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
fig_n = 'fld_traj_zc_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'$\Delta\chi(\phi=\phi_p)>0$',r'$\Delta\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\phi$', r'$\Delta\dot{\phi}$']
fig.suptitle(f_title)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1].set_xlabel(x_lab[-1])
	#ax[0,i].set_title(s_title[i])

print('traj_cut_bin_full = ', traj_cut_bin_full)
#bin_select = [0,2]
i = 0
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[0].fill_between(np.log(en[0,::sl,a_i]), (Delta_fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T, (Delta_fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T, alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[1].fill_between(np.log(en[0,::sl,a_i]), ((Delta_dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T)/en[0,::sl,a_i]**3, ((Delta_dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T)/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
#bin_select = [0,1]
ns = 8  # number of sampled trajectories
i = 0
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			if (traj_cut_bins_i[l][j+1] < len(cut[l])): # replace nlat with number of lattice sites in this cut
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld[fld_i[i,0],fld_i[i,1],:,ind_u], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld[fld_i[i,0],fld_i[i,1],:,ind_l], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_u]/en_bl[::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_l]/en_bl[::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[0].plot(np.log(en[0,::sl,a_i]), (fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - fld[fld_i[i,0],fld_i[i,1],:,ind_s]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_s]/en_bl[::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax:
	axis.legend(loc=2)
if SAVE_FIGS:
	plt.savefig(fig_n)

# Plot 5: Plot of \Delta\chi and \Delta\dot{\chi} cut by \Delta\chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
fig_n = 'fld_traj_zc_plt' + str(nfig) + run_ident[0] + '.png'
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'$\Delta\chi(\phi=\phi_p)>0$',r'$\Delta\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\chi$', r'$\Delta\dot{\chi}$']
fig.suptitle(f_title)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1].set_xlabel(x_lab[-1])
	#ax[0,i].set_title(s_title[i])

print('traj_cut_bin_full = ', traj_cut_bin_full)
#bin_select = [0,2]
i = 1
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[0].fill_between(np.log(en[0,::sl,a_i]), (Delta_fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T, (Delta_fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T, alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[1].fill_between(np.log(en[0,::sl,a_i]), ((Delta_dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T)/en[0,::sl,a_i]**3, ((Delta_dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T)/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
#bin_select = [0,1]
ns = 8  # number of sampled trajectories
i = 1
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			if (traj_cut_bins_i[l][j+1] < len(cut[l])): # replace nlat with number of lattice sites in this cut
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld[fld_i[i,0],fld_i[i,1],:,ind_u], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[0].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld[fld_i[i,0],fld_i[i,1],:,ind_l], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_u]/en_bl[::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_l]/en_bl[::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[0].plot(np.log(en[0,::sl,a_i]), (fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - fld[fld_i[i,0],fld_i[i,1],:,ind_s]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[1].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]/en[0,::sl,a_i]**3 - dfld[fld_i[i,0],fld_i[i,1],:,ind_s]/en_bl[::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax:
	axis.legend(loc=2)
if SAVE_FIGS:
	plt.savefig(fig_n)

if SHOW_FIGS:
	plt.show()
