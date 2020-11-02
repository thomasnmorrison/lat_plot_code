# fld_traj_zeta_condition.py

# Script to plot field and momentum trajectories conditioned on \zeta.
# Plot 1: Field and momentum trajectory contour plots (contoured by \Delta\zeta_{end}) binned by \Delta\zeta_{end} 
#         with sample trajectories vs \alpha
# Plot 2: Field and momentum trajectory contour plots binned by \Delta\zeta_{end} with sample trajectories vs \alpha

# to do: plot sampled trajectories for each bin, top, bottom, evenly spaced samples
# to do: sharex and remove redundant axis labels and ticks
# to do: sort cut \Delta fields

# Testing notes:

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from plot_param import *

title_fs = 20
stitle_fs = 14
x_lab_fs = 14; y_lab_fs = 14

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

# Apply cuts on \chi
# cut_i: index of slice where division into +- \chi trajectories will be made
# cut: tuple of (ind for + \chi traj, ind for - \chi traj), cast to an np array
#cut_i = np.searchsorted(np.sort(en[0,::sl,phi_i]), phi_p-phi_w)
#cut_i = np.argmin(np.abs(en[0,::sl,phi_i] - (phi_p-phi_w)))
cut_i = np.argmin(np.abs(en[0,::sl,phi_i] - (phi_p+0.05)))
cut = (np.argwhere(np.where(chi[0,cut_i,:]>=0.,1,0)), np.argwhere(np.where(chi[0,cut_i,:]<0.,1,0)))
cut = np.array(cut)
for i, c in enumerate(cut):
	cut[i] = c.flatten()

# Sort data by \Delta\zeta_{end}
# Dz_sort: sorted values of \Delta\zeta_{end}
# Dz_sort_i: ind to sort \Delta\zeta_{end}
Dz_sort = np.sort((zeta[0]-zeta_bl)[-1,:],axis=-1)       # making this for baseline + one run
Dz_sort_i = np.argsort((zeta[0]-zeta_bl)[-1,:],axis=-1)  # making this for baseline + one run

# Sort cut data by \Delta\zeta_{end}
# Dz_cut_sort: tuple of (sorted \Delta\zeta_{end} of + \chi traj, sorted \Delta\zeta_{end} of - \chi traj)
# Dz_cut_sort_i: tuple of (ind to sort \Delta\zeta_{end} of + \chi traj, ind to sort \Delta\zeta_{end} of - \chi traj)
# Dz_cut_sort_i: formatted []
# to do: using 
Dz_cut_sort = ()
Dz_cut_sort_i = ()
for c in cut:
	Dz_cut_sort = Dz_cut_sort + (np.sort((zeta[0]-zeta_bl)[-1,c],axis=-1),)
	Dz_cut_sort_i = Dz_cut_sort_i + (np.argsort((zeta[0]-zeta_bl)[-1,c],axis=-1),)

# Define trajectory bins, bin by \Delta\zeta_{end}
# traj_bin_label: list of bi labels for plot legends
sig_Dz = np.std((zeta-zeta_bl)[0,-1,:], ddof=-1)   # \Delta\zeta_{end} std
traj_bins = np.array([-5,5])*sig_Dz             # trajector bin edges in terms of std of \Delta\zeta
traj_bins_i = np.zeros(len(traj_bins)+2, dtype=np.int64)
traj_bins_i[-1] = nlat-1                          # outside bin edge
traj_bins_i[1:-1] = np.searchsorted(Dz_sort, traj_bins) # array of indices in Dz_sort_i that correspond to the bins in traj_bins
traj_bin_label = ['']*(len(traj_bins)+1)
traj_bin_label[0] = r'$<{:03.2f}\sigma$'.format(traj_bins[0]/sig_Dz)
for i in range(1,len(traj_bins)):
	traj_bin_label[i] = r'$({:03.2f},{:03.2f})\sigma$'.format(traj_bins[i-1]/sig_Dz, traj_bins[i]/sig_Dz)
traj_bin_label[-1] = r'>{0}\sigma'.format(traj_bins[-1]/sig_Dz)


# Define cut trajectory bins
# traj_cut_bins_i: tuple of (ind of cut for bin edges of \Delta\zeta_{end} for + \chi traj, ind of cut for bin edges of \Delta\zeta_{end} for + \chi traj), 
traj_cut_bins_i = ()
for i, c in enumerate(cut):
	cut_bins = np.zeros(len(traj_bins)+2, dtype=np.int64)
	cut_bins[-1] = len(c)-1
	cut_bins[1:-1] = np.searchsorted(Dz_cut_sort[i], traj_bins)
	#cut_bins[1:-1] = np.where((np.searchsorted(Dz_cut_sort[i], traj_bins)<len(c)), np.searchsorted(Dz_cut_sort[i], traj_bins), len(c)-1)
	traj_cut_bins_i = traj_cut_bins_i + (cut_bins,)
traj_cut_bin_full = np.array([[False]*(len(traj_bins)+1)]*len(cut))

# Define bin contour variables
cont = np.array([0., 0.05, 0.25, 0.75, 0.95, 1.])  # contour levels
#cont = np.array([0., 0.25, 0.75, 1.])  # contour levels
phi_bl_cz = np.zeros((len(traj_bins)+1, len(cont), nl))      # Formatted [bin, contour, time]
dphi_bl_cz = np.zeros((len(traj_bins)+1, len(cont), nl))     # Formatted [bin, contour, time]
phi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))         # Formatted [bin, contour, time]
dphi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))        # Formatted [bin, contour, time]
Delta_phi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))   # Formatted [bin, contour, time]
Delta_dphi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))  # Formatted [bin, contour, time]
fld_bl_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))      # Formatted [fld, bin, contour, time]
dfld_bl_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))     # Formatted [fld, bin, contour, time]
fld_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))         # Formatted [fld, bin, contour, time]
dfld_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))        # Formatted [fld, bin, contour, time]
Delta_fld_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))   # Formatted [fld, bin, contour, time]
Delta_dfld_c = np.zeros((nfld, len(traj_bins)+1, len(cont), nl))  # Formatted [fld, bin, contour, time]

if (chi_bl_f != ''):
	chi_bl_cz = np.zeros((len(traj_bins)+1, len(cont), nl))       # Formatted [bin, contour, time]
	dchi_bl_cz = np.zeros((len(traj_bins)+1, len(cont), nl))      # Formatted [bin, contour, time]
	chi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))          # Formatted [bin, contour, time]
	dchi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))         # Formatted [bin, contour, time]
	Delta_chi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))    # Formatted [bin, contour, time]
	Delta_dchi_cz = np.zeros((len(traj_bins)+1, len(cont), nl))   # Formatted [bin, contour, time]

# Define bin contours on cut data
# Formatted [[fld], cut, bin, cont, t]
# Delta_zeta_cut_cz: formatted [cut, bin, cont, t]
print('np.shape(f1_bl_i) = ', np.shape(f1_bl_i))
print('(nfld,2,len(cut), len(traj_bins)+1, len(cont), nl)) = ', (nfld,2,len(cut), len(traj_bins)+1, len(cont), nl))
fld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl)) 
dfld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_fld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_dfld_cut_cz = np.zeros((nfld, 2, len(cut), len(traj_bins)+1, len(cont), nl))
Delta_zeta_cut_cz = np.zeros((len(cut), len(traj_bins)+1, len(cont), nl))

# Find bin contours
for j in range(1,len(traj_bins_i)):
	k = traj_bins_i[j] - traj_bins_i[j-1]           # number of trajectories in bin
	print('j = ', j)
	print('traj_bins_i[j-1] = ', traj_bins_i[j-1])
	print('traj_bins_i[j] =', traj_bins_i[j])
	ind = Dz_sort_i[traj_bins_i[j-1]:traj_bins_i[j]]  # indecies of trajectories in this bin
	Dphi_bin_sort = np.sort((phi[0,]-phi_bl)[:,ind], axis=-1).T       # Formatted [traj, time]
	Ddphi_bin_sort = np.sort((dphi[0,]-dphi_bl)[:,ind], axis=-1).T    # Formatted [traj, time]
	print('np.shape(Dphi_bin_sort) = ', np.shape(Dphi_bin_sort))
	if (chi_bl_f != ''):
		Dchi_bin_sort = np.sort((chi[0,]-chi_bl)[:,ind], axis=-1).T     # Formatted [traj, time]
		Ddchi_bin_sort = np.sort((dchi[0,]-dchi_bl)[:,ind], axis=-1).T  # Formatted [traj, time]
	for i, c in enumerate(cont):
		if (k>0):
			cont_ind = int(c*(k-1))                    # index of contour
			print('k = ', k)
			print('cont_ind = ', cont_ind)
			#indz = Dz_sort_i[traj_bins_i[j]+cont_ind]  # index of contour by sorted \Delta\zeta_{end}
			indz = Dz_sort_i[traj_bins_i[j-1]+cont_ind]  # index of contour by sorted \Delta\zeta_{end}
			phi_bl_cz[j-1,i,:] = phi_bl[:,indz].T
			dphi_bl_cz[j-1,i,:] = dphi_bl[:,indz].T
			phi_cz[j-1,i,:] = phi[0,:,indz].T
			dphi_cz[j-1,i,:] = dphi[0,:,indz].T
			Delta_phi_cz[j-1,i,:] = (phi[0,:,indz]-phi_bl[:,indz]).T
			Delta_dphi_cz[j-1,i,:] = (dphi[0,:,indz]-dphi_bl[:,indz]).T
			Delta_fld_c[0,j-1,i] = Dphi_bin_sort[cont_ind,:]
			Delta_dfld_c[0,j-1,i] = Ddphi_bin_sort[cont_ind,:]
			if (chi_bl_f != ''):
				chi_bl_cz[j-1,i,:] = chi_bl[:,indz].T
				dchi_bl_cz[j-1,i,:] = dchi_bl[:,indz].T
				chi_cz[j-1,i,:] = chi[0,:,indz].T
				dchi_cz[j-1,i,:] = dchi[0,:,indz].T
				Delta_chi_cz[j-1,i,:] = (chi[0,:,indz]-chi_bl[:,indz]).T
				Delta_dchi_cz[j-1,i,:] = (dchi[0,:,indz]-dchi_bl[:,indz]).T
				Delta_fld_c[1,j-1,i] = Dchi_bin_sort[cont_ind,:]
				Delta_dfld_c[1,j-1,i] = Ddchi_bin_sort[cont_ind,:]

# Find cut bin contours
# to do: check if I need to +1 to the index end when calculating ind, it looks like I'm cutting the last trajectory
for i, c in enumerate(cut):
	for j in range(1,len(traj_bins_i)):
		n_traj = traj_cut_bins_i[i][j] - traj_cut_bins_i[i][j-1]   # Find number of trajectories in bin for this cut	
		ind = c[Dz_cut_sort_i[i][traj_cut_bins_i[i][j-1]:traj_cut_bins_i[i][j]]]  # Find indices of trajectories in this bin for this cut
		print('i = ', i)
		print('n_traj = ', n_traj)
		print('len(ind) = ', len(ind))
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
		zeta_sort = np.sort((zeta[0]-zeta_bl)[:,ind], axis=-1).T    # Formatted [traj, time]
		for k, d in enumerate(cont):
			if (n_traj > 0):
				cont_ind = int(d*(n_traj - 1))                # index of contour
				Delta_zeta_cut_cz[i,j-1,k,:] = zeta_sort[cont_ind,:]

# Make plots
nfig = 0

# Plot 1: Field and momentum trajectory contour plots (contoured by \Delta\zeta_{end}) binned by \Delta\zeta_{end} 
#         with sample trajectories vs \alpha
# to do: select colours
# to do: select sample trajectories to be plotted
# to do: plot sample trajectories
nfig += 1
print('In plot: ', nfig)
nr = 4; nc =1
fig, ax = plt.subplots(nrows=4 , ncols=1 , sharex=False)
f_title = r''
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\phi$', r'$\Delta\Phi_phi$',r'$\Delta\chi$', r'$\Delta\Phi_chi$']
#fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i].set_xlabel(x_lab[i])
	ax[i].set_ylabel(y_lab[i])

# loop over bins
for j in range(0,len(traj_bins)+1):
	for i in range(0,int(len(cont)/2)):
		ax[0].fill_between(np.log(en[0,::sl,a_i]), (phi_cz[j,i,:]-en[0,::sl,phi_i]).T, (phi_cz[j,-1-i,:]-en[0,::sl,phi_i]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
		ax[1].fill_between(np.log(en[0,::sl,a_i]), (dphi_cz[j,i,:]-en[0,::sl,dphi_i]).T, (dphi_cz[j,-1-i,:]-en[0,::sl,dphi_i]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
		ax[2].fill_between(np.log(en[0,::sl,a_i]), (chi_cz[j,i,:]-en[0,::sl,chi_i]).T, (chi_cz[j,-1-i,:]-en[0,::sl,chi_i]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
		ax[3].fill_between(np.log(en[0,::sl,a_i]), (dchi_cz[j,i,:]-en[0,::sl,dchi_i]).T, (dchi_cz[j,-1-i,:]-en[0,::sl,dchi_i]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
	# Plot sample trajectories for bin
	#ax[0].plot(np.log(en[0,::sl,a_i]),)

# Plot 2: Field and momentum trajectory contour plots binned by \Delta\zeta_{end} with sample trajectories vs \alpha
# to do: select sample trajectories to be plotted
# to do: plot sample trajectories
nfig += 1
print('In plot: ', nfig)
nr = 4; nc =1
fig, ax = plt.subplots(nrows=4 , ncols=1 , sharex=True)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\phi$', r'$\Delta\Pi_\phi$',r'$\Delta\chi$', r'$\Delta\Pi_\chi$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
ax[-1].set_xlabel(x_lab[-1])

# loop over bins
for j in range(0,len(traj_bins)+1):
	for i in range(0,int(len(cont)/2)):
		for k in range(0,nfld):
			ax[2*k].fill_between(np.log(en[0,::sl,a_i]), (Delta_fld_c[k,j,i,:]).T, (Delta_fld_c[k,j,-1-i,:]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
			ax[2*k+1].fill_between(np.log(en[0,::sl,a_i]), (Delta_dfld_c[k,j,i,:]).T, (Delta_dfld_c[k,j,-1-i,:]).T, alpha=0.25, color=c_ar[j%len(c_ar)])
		# Plot bounding trajectories
for j in [0,2]:
	n_traj = traj_bins_i[j+1] - traj_bins_i[j]           # number of trajectories in bin
	if (traj_bins_i[j+1] < nlat):
		ind_u = Dz_sort_i[traj_bins_i[j+1]]
	else:
		ind_u = Dz_sort_i[-1]
	if (traj_bins_i[j] < nlat):
		ind_l = Dz_sort_i[traj_bins_i[j]]
	else:
		ind_l = Dz_sort_i[-1]
	print('ind_u, ind_l = ', ind_u, ind_l)
	ax[0].plot(np.log(en[0,::sl,a_i]), phi[0,:,ind_u]-phi_bl[:,ind_u], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[0].plot(np.log(en[0,::sl,a_i]), phi[0,:,ind_l]-phi_bl[:,ind_l], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[1].plot(np.log(en[0,::sl,a_i]), dphi[0,:,ind_u]-dphi_bl[:,ind_u], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[1].plot(np.log(en[0,::sl,a_i]), dphi[0,:,ind_l]-dphi_bl[:,ind_l], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[2].plot(np.log(en[0,::sl,a_i]), chi[0,:,ind_u]-chi_bl[:,ind_u], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[2].plot(np.log(en[0,::sl,a_i]), chi[0,:,ind_l]-chi_bl[:,ind_l], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[3].plot(np.log(en[0,::sl,a_i]), dchi[0,:,ind_u]-dchi_bl[:,ind_u], alpha=0.75, color=c_ar[j%len(c_ar)])
	ax[3].plot(np.log(en[0,::sl,a_i]), dchi[0,:,ind_l]-dchi_bl[:,ind_l], alpha=0.75, color=c_ar[j%len(c_ar)])
			# Plot sampled trajectories
	# Plot sample trajectories for bin
	#ax[0].plot(np.log(en[0,::sl,a_i]),)

# Plot 3:
nfig += 1
print('In plot: ', nfig)
nr = 4; nc =1
fig, ax = plt.subplots(nrows=4 , ncols=1 , sharex=True)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'',r'']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\phi$', r'$\Delta\Pi_\phi$',r'$\Delta\chi$', r'$\Delta\Pi_\chi$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i].set_ylabel(y_lab[i])
#ax[0].set_xticks([])
ax[-1].set_xlabel(x_lab[-1])
ax[0].plot(np.log(en[0,::sl,a_i]), chi[0][:,cut[0][::1000]], c='r')
ax[0].plot(np.log(en[0,::sl,a_i]), chi[0][:,cut[1][::1000]], c='b')
ax[0].plot(np.log(en[0,::sl,a_i]), chi[0][:,::1000], c='k', alpha=0.15)
for i in range(0,int(len(cont)/2)):
	ax[2].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[3,0],fld_i[3,1],0,0,i,:]).T, (fld_cut_cz[fld_i[3,0],fld_i[3,1],0,0,-1-i,:]).T, alpha=0.25, color='r')
	ax[1].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[3,0],fld_i[3,1],1,0,i,:]).T, (fld_cut_cz[fld_i[3,0],fld_i[3,1],1,0,-1-i,:]).T, alpha=0.25, color='b')
print('np.log(en[0,cut_i*sl,a_i]) = ', np.log(en[0,cut_i*sl,a_i]))


# Plot 4: Plot of fields and momenta cut by \chi(\phi=\phi_p-\phi_w) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
# to do: 
nfig += 1
print()
print('In plot: ', nfig)
nr = 4; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'Where $\chi(\phi=\phi_p)>0$',r'Where $\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\phi-\langle\phi\rangle$', r'$\dot{\phi}-\langle\dot{\phi}\rangle$',
r'$\chi-\langle\chi\rangle$', r'$\dot{\chi}-\langle\dot{\chi}\rangle$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i,0].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1,i].set_xlabel(x_lab[-1])
	ax[0,i].set_title(s_title[i])

print('traj_cut_bin_full = ', traj_cut_bin_full)
bin_select = [0,2]
for i in range(0,nfld):
	for j in range(0,len(traj_bins)+1):# bin_select
		for l in range(0,len(cut)):
		#l=0
			if (traj_cut_bin_full[l,j]):
				for k in range(1,int(len(cont)/2)):		
					line = ax[2*i,l].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], alpha=0.15, color=c_ar[j%len(c_ar)])
					line1 = ax[2*i+1,l].fill_between(np.log(en[0,::sl,a_i]), ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
				line.set_label(traj_bin_label[j])
				line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
bin_select = [0,2]
ns = 8  # number of sampled trajectories
print()
for i in range(0,nfld):
	for j in bin_select:
		for l in range(0,len(cut)):
			if (traj_cut_bin_full[l,j]):
				print('i,j,l = ', i,j,l)
				print('np.shape(cut[l]) = ', np.shape(cut[l]))
				print('np.shape(traj_cut_bins_i[l][j]) = ', np.shape(traj_cut_bins_i[l][j]))
				print('traj_cut_bins_i[l] = ', traj_cut_bins_i[l])
				print('Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]] = ', Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]])
				if (traj_bins_i[j+1] < nlat):
					ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
				else:
					ind_u = cut[l][Dz_cut_sort_i[l][-1]]
				if (traj_bins_i[j+1] < nlat):
					ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
				else:
					ind_l = cut[l][Dz_cut_sort_i[l][-1]]
				if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
					ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
				else:
					ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
				# Plot bin edge trajectories
				ax[2*i,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				ax[2*i,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
				print('ind_s =', ind_s)
				print('np.shape(ind_s) = ', np.shape(ind_s))
				ax[2*i,l].plot(np.log(en[0,::sl,a_i]), ((fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]) - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
				ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), ((dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

#line.set_label('test')
for axis in ax.flatten():
	axis.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	axis.set_axisbelow(True)
	axis.grid()
	axis.set_xmargin(0)
	axis.legend(fontsize=6, loc=1)

# Plot 5: Plot of \zeta cut by \chi(\phi=\phi_p-\phi_w) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.

nfig += 1
print()
print('In plot: ', nfig)
nr = 1; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=True)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'Where $\chi(\phi=\phi_p)>0$',r'Where $\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\Delta\zeta$']
fig.suptitle(f_title, fontsize = title_fs)
ax[0].set_ylabel(y_lab[0])
for i in range(0,nc):
	ax[i].set_xlabel(x_lab[-1])
	ax[i].set_title(s_title[i])

for axis in ax.flatten():
	axis.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	axis.set_axisbelow(True)
	axis.grid()
	axis.set_xmargin(m=0.)

# Plot contours
bin_select = [0,1]  # select bins to plot contours
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[l].fill_between(np.log(en[0,::sl,a_i]), (Delta_zeta_cut_cz[l,j,k,:]).T, (Delta_zeta_cut_cz[l,j,-1-k,:]).T, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])

# Plot trajectories
bin_select = [0,1]  # select bins to plot sampled trajectories
ns = 10  # number of sampled trajectories
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			#ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			#ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			if (traj_bins_i[j+1] < nlat):
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
			if (traj_bins_i[j+1] < nlat):
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[l].plot(np.log(en[0,::sl,a_i]), (zeta[0]-zeta_bl)[:,ind_u], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[l].plot(np.log(en[0,::sl,a_i]), (zeta[0]-zeta_bl)[:,ind_l], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			# Plot sampled trajectories in bin
			ax[l].plot(np.log(en[0,::sl,a_i]), ((zeta[0]-zeta_bl)[:,ind_s]), lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax.flatten():
	axis.legend(fontsize=6, loc=2)

# Plot 6: Plot of \phi and \Pi_\phi cut by \chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.

nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'Where $\chi(\phi=\phi_p)>0$',r'Where $\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\phi-\langle\phi\rangle$', r'$\dot{\phi}-\langle\dot{\phi}\rangle$',
r'$\chi-\langle\chi\rangle$', r'$\dot{\chi}-\langle\dot{\chi}\rangle$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i,0].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1,i].set_xlabel(x_lab[-1])
	ax[0,i].set_title(s_title[i])
for axis in ax.flatten():
	axis.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	axis.set_axisbelow(True)
	axis.grid()
	axis.set_xmargin(0)

print('traj_cut_bin_full = ', traj_cut_bin_full)
bin_select = [0,2]
i = 0
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[2*i,l].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[2*i+1,l].fill_between(np.log(en[0,::sl,a_i]), ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
bin_select = [0,2]
ns = 8  # number of sampled trajectories
print()
i = 0
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			#ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			#ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			if (traj_bins_i[j+1] < nlat):
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
			if (traj_bins_i[j+1] < nlat):
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[2*i,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[2*i,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[2*i,l].plot(np.log(en[0,::sl,a_i]), ((fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]) - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[2*i+1,l].plot(np.log(en[0,::sl,a_i]), ((dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax.flatten():
	axis.legend(fontsize=6, loc=1)

# Plot 6: Plot of \chi and \Pi_\chi cut by \chi(\phi=\phi_p) >(<) 0.
#         Trajectories in each cut are binned by \Delta\zeta_{end}.
#         Each bin is plotted by contours and sampled trajectories.
nfig += 1
print()
print('In plot: ', nfig)
nr = 2; nc = len(cut)
fig, ax = plt.subplots(nrows=nr , ncols=nc , sharex=True, sharey=False)
f_title = r'Trajectories Binned by $\Delta\zeta_{\mathrm{end}}$'
s_title = [r'Where $\chi(\phi=\phi_p)>0$',r'Where $\chi(\phi=\phi_p)>0$']
x_lab = [r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$', r'$\mathrm{ln}(a)$']
y_lab = [r'$\chi-\langle\chi\rangle$', r'$\dot{\chi}-\langle\dot{\chi}\rangle$']
fig.suptitle(f_title, fontsize = title_fs)
for i in range(0,nr):
	ax[i,0].set_ylabel(y_lab[i])
for i in range(0,nc):
	ax[-1,i].set_xlabel(x_lab[-1])
	ax[0,i].set_title(s_title[i])
for axis in ax.flatten():
	axis.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
	axis.set_axisbelow(True)
	axis.grid()
	axis.set_xmargin(0)

print('traj_cut_bin_full = ', traj_cut_bin_full)
bin_select = [0,2]
i = 1
for j in range(0,len(traj_bins)+1):# bin_select
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			for k in range(0,int(len(cont)/2)):		
				line = ax[0,l].fill_between(np.log(en[0,::sl,a_i]), (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], (fld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], alpha=0.15, color=c_ar[j%len(c_ar)])
				line1 = ax[1,l].fill_between(np.log(en[0,::sl,a_i]), ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ((dfld_cut_cz[fld_i[nfld+i,0],fld_i[nfld+i,1],l,j,-1-k,:]).T - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, alpha=0.15, color=c_ar[j%len(c_ar)])
			line.set_label(traj_bin_label[j])
			line1.set_label(traj_bin_label[j])

# Plot over contours for select bins the edge trajecories and ampled trajectories
bin_select = [0,2]
ns = 8  # number of sampled trajectories
print()
i = 1
for j in bin_select:
	for l in range(0,len(cut)):
		if (traj_cut_bin_full[l,j]):
			#ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			#ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			if (traj_bins_i[j+1] < nlat):
				ind_u = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j+1]]]
			else:
				ind_u = cut[l][Dz_cut_sort_i[l][-1]]
			if (traj_bins_i[j+1] < nlat):
				ind_l = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]]]
			else:
				ind_l = cut[l][Dz_cut_sort_i[l][-1]]
			if ((traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns > 0):
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]:(traj_cut_bins_i[l][j+1]-traj_cut_bins_i[l][j])//ns]]
			else:
				ind_s = cut[l][Dz_cut_sort_i[l][traj_cut_bins_i[l][j]:traj_cut_bins_i[l][j+1]]]
			# Plot bin edge trajectories
			ax[0,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[0,l].plot(np.log(en[0,::sl,a_i]), fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]], ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_u] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
			ax[1,l].plot(np.log(en[0,::sl,a_i]), (dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_l] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3, ls='-', alpha=0.75, color=c_ar[j%len(c_ar)])
				# Plot sampled trajectories in bin
			ax[0,l].plot(np.log(en[0,::sl,a_i]), ((fld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s]) - fld0[fld_i[nfld+i,0],fld_i[nfld+i,1]]).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])
			ax[1,l].plot(np.log(en[0,::sl,a_i]), ((dfld[fld_i[nfld+i,0],fld_i[nfld+i,1],:,ind_s] - dfld0[fld_i[nfld+i,0],fld_i[nfld+i,1]])/en[0,::sl,a_i]**3).T, lw=0.75, ls='-.', alpha=0.5, color=c_ar[j%len(c_ar)])

for axis in ax.flatten():
	axis.legend(fontsize=6, loc=1)

plt.show()
