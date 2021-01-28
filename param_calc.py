# to do: give an mp and calculate a \phi_w and \phi_p for equivalent strength

import numpy as np
n = 2**8
mp = np.zeros(n)
l = np.zeros(n)
chi_w = 0.02
mp[n-1] = 40.
H0 = 3.6
delta = np.exp(mp[n-1]/H0)/n
for i in range(1,n):
	mp[n-i-1] = H0*np.log(np.exp(mp[n-i]/H0)-delta)
mp2 = -mp**2

#for i in range(0,n):
#	print('i, mp2, l = {0}, {1}, {2}'.format(i, mp2[i], l[i]))


# linear spaceing of |m_p|
m = 2**3
mp_lin = np.zeros(n)
mp_lin[m-1] = mp[0]
print('mp2[0] = ', mp2[0])
mp_lin = np.linspace(0, mp[0], m, endpoint=False)
mp2_lin = -mp_lin**2
l = -mp2_lin/chi_w**2

f = open('pot_params3.txt', 'w')
for i in range(0,m):
	f.write('i, mp2, l = {0}, {1}, {2} \n'.format(i, mp2_lin[i], l[i]))
l = -mp2/chi_w**2
for i in range(0,n):
	f.write('i, mp2, l = {0}, {1}, {2} \n'.format(i, mp2[i], l[i]))
f.close()

ind = 40-1-m
n1 = 2**2+1
step = 2**5
mp_fixed = mp[ind]
l_fixed = l[ind]
phi_p = np.zeros(n1)
phi_w = np.zeros(n1)

phi_p[0] = 8.5
phi_w[0] = 0.1
for i in range(0,n1):
	phi_w[i] = mp[ind + i*step]*phi_w[0]/mp_fixed
	phi_p[i] = phi_p[0] + phi_w[0] - phi_w[i]

f = open('pot_params4.txt', 'w')
for i in range(0,n1):
	f.write('i, mp2, l, phi_p, phi_w = {0}, {1}, {2}, {3}, {4} \n'.format(i, mp2[ind], l[ind], phi_p[i], phi_w[i]))
f.close()
