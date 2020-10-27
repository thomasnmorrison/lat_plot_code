# fld_traj_zeta_condition.py

# Script to plot field and momentum trajectories conditioned on \zeta.
# Plot 1: Field and momentum trajectory contour plots (contoured by \Delta\zeta_{end}) binned by \Delta\zeta_{end} 
#         with sample trajectories vs \alpha
# Plot 2: Field and momentum trajectory contour plots binned by \Delta\zeta_{end} with sample trajectories vs \alpha

# to do: plot sampled trajectories for each bin, top, bottom, evenly spaced samples
# to do: bin positive and negatice \chi separately
# to do: sharex and remove redundant axis labels and ticks

# to do: apply cut (use argwhere), sort zeta on cut data, 

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

print('np.shape(phi_bl) = ', np.shape(phi_bl))
print('np.shape(phi) = ', np.shape(phi))

# Apply cuts on \chi

# Sort data by \Delta\zeta_{end}
Dz_sort = np.sort((zeta[0]-zeta_bl)[-1,:],axis=-1)    # making this for baseline + one run
Dz_sort_i = np.argsort((zeta[0]-zeta_bl)[-1,:],axis=-1)  # making this for baseline + one run

# Sort data by \Delta\zeta_

# Define trajectory bins, bin by \Delta\zeta_{end}
sig_Dz = np.std((zeta-zeta_bl)[0,-1,:], ddof=-1)   # \Delta\zeta_{end} std
traj_bins = np.array([-4,3])*sig_Dz             # trajector bin edges in terms of std of \Delta\zeta
traj_bins_i = np.zeros(len(traj_bins)+2, dtype=np.int64)
traj_bins_i[-1] = nlat-1                          # outside bin edge
traj_bins_i[1:-1] = np.searchsorted(Dz_sort, traj_bins) # array of indices in Dz_sort_i that correspond to the bins in traj_bins

# Define bin contour variables
cont = np.array([0., 0.05, 0.25, 0.75, 0.95, 1.])  # contour levels
cont = np.array([0., 0.25, 0.75, 1.])  # contour levels
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
	Delta_chi_cpn = np.zeros((2, len(traj_bins)+1, len(cont), nl))  # Formatted [fld, bin, contour, time], for positive/negative \chi trajectories
	Delta_dchi_cpn = np.zeros((2, len(traj_bins)+1, len(cont), nl))  # Formatted [fld, bin, contour, time], for positive/negative \Pi_\chi trajectories

print('np.shape(phi_bl_cz) = ', np.shape(phi_bl_cz))
print('np.shape(Delta_fld_c) = ', np.shape(Delta_fld_c))

# Find bin contours
# to do: choose a time slice to cut \chi and \Pi_\chi into positive and negative branches
# to do: divide Dz_sort_i into an array for positve \chi and an array for negative \chi
# to do: divede trajectories based on positive or negative \chi, then bin based on \zeta_{end}
for j in range(1,len(traj_bins_i)):
	k = np.mod(traj_bins_i[j], nlat)-np.mod(traj_bins_i[j-1], nlat)           # number of trajectories in bin
	print('j, np.mod(traj_bins_i[j-1],nlat), np.mod(traj_bins_i[j],nlat) =', j, np.mod(traj_bins_i[j-1],nlat), np.mod(traj_bins_i[j],nlat))
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
	ind_u = Dz_sort_i[traj_bins_i[j+1]]
	ind_l = Dz_sort_i[traj_bins_i[j]]
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

plt.show()
