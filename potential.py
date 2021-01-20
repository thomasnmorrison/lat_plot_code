# potential.py

# Module containing functions for evaluating the potential and its derivatives

### Include packages
import numpy as np

### Potential parameters
mpl = np.ones(1)				# M_Pl mass
m2_phi = np.ones(1)			# phi mass^2
m2_inf = np.ones(1)			# asymptotic chi mass^2
lambda_chi = np.ones(1)	# stabilizing quartic term
phi_p = np.ones(1)			# centre of instability
phi_w = np.ones(1)			# half width of instability
m2_p = np.ones(1)				# instability chi mass^2

### Input functions

# Initialize mpl
def init_mpl(mpl_in):
	mpl[0] = mpl_in
	return

# Initialize potential parameters
def init_param(phi_p_in, phi_w_in, m2_p_in, m2_phi_in=1., m2_inf_in=1., lambda_chi_in=10.):
	phi_p[0] = phi_p_in
	phi_w[0] = phi_w_in
	m2_p[0] = m2_p_in
	m2_phi[0] = m2_phi_in
	m2_inf[0] = m2_inf_in
	lambda_chi[0] = lambda_chi_in
	return

### Potential Functions

# phi dependent portion of potential
def V_0_phi(f1, f2, mpl_n=False):
	V_0_phi = 0.5*m2_phi*f1**2
	if mpl_n==True:
		V_0_phi = V_0_phi/mpl[0]**2
	return V_0_phi

# chi dependent portion of potential
def V_0_chi(f1, f2, mpl_n=False):
	V_0_chi = 0.5*m2_inf*f2**2 + 0.25*lambda_chi*f2**4
	if mpl_n==True:
		V_0_chi = V_0_chi/mpl[0]**2
	return V_0_chi

# phi-chi interaction potential
def V_int(f1, f2, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	V_int = 0.5*(-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0]))*( m2_p[0]-m2_inf[0] + g2*(f1-phi_p[0])**2 - g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4) * f2**2
	if mpl_n==True:
		V_int = V_int/mpl[0]**2
	return V_int

# chi effective mass term (2nd chi partial derivative)
def m2eff_chi(f1, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	m2eff = (-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0]))*( m2_p[0]-m2_inf[0] + g2*(f1-phi_p[0])**2 - g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4)+m2_inf[0]
	if mpl_n==True:
		m2eff = m2eff/mpl[0]**2
	return m2eff

# chi at the minimum of the trough
def chi_min(f1):
        g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
        if (np.abs(f1-phi_p)>phi_w):
                chi_min = 0.
        else:
                chi_min = np.sqrt((-m2_p[0]+m2_inf[0]-g2*(f1-phi_p[0])**2) / (- g2**2/(2.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4))
        return chi_min
                
# Full potential, with baseline option to ignore interaction term
def V(f1, f2, mpl_n=False, baseline=False):
	if baseline==True:
		V = V_0_phi(f1, f2) + V_0_chi(f1, f2)
	if baseline==False:
		V = V_0_phi(f1, f2) + V_0_chi(f1, f2) + V_int(f1, f2)
	if mpl_n==True:
		V= V/mpl[0]**2
	return V

# Quadratic portion of the baseline potential
def V_0(f1, f2, mpl_n=False):
        V_0 = 0.5*m2_phi*f1**2 + 0.5*m2_inf*f2**2
        if mpl_n==True:
                V_0 = V_0/mpl[0]**2
        return V_0

### Potential Derivatives

# phi partial derivative of V_0_phi
def dV_0_phi(f1, f2, mpl_n=False):
	dV_0_phi = m2_phi*f1
	if mpl_n==True:
		dV_0_phi = dV_0_phi/mpl[0]**2
	return dV_0_phi

# chi partial derivative of V_0_chi
def dV_0_chi(f1, f2, mpl_n=False):
	dV_0_chi = m2_inf*f2 + lambda_chi*f2**3
	if mpl_n==True:
		dV_0_chi = dV_0_chi/mpl[0]**2
	return dV_0_chi

# partial derivatives of V_int, ind=1 for phi partial, ind=2 for chi partial
def dV_int(f1, f2, ind, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	c_inf = np.sqrt(2.*(m2_inf[0]-m2_p[0])/g2)
	if ind==1:
		dV_int = 0.5*(-0.5*np.sign(-f1+phi_p-c_inf)-0.5*np.sign(f1-phi_p-c_inf))*(2.*g2*(f1-phi_p) - g2**2/((m2_inf-m2_p))*(f1-phi_p)**3) * f2**2
	if ind==2:
		dV_int = (-0.5*np.sign(-f1+phi_p-c_inf)-0.5*np.sign(f1-phi_p-c_inf))*( m2_p-m2_inf + g2*(f1-phi_p)**2 - g2**2/(4.*(m2_inf-m2_p))*(f1-phi_p)**4) * f2
	if mpl_n==True:
		dV_int = dV_int/mpl[0]**2
	return dV_int

# partial derivatives of V, ind=1 for phi partial, ind=2 for chi partial
def dV(f1, f2, ind, mpl_n=False):
	if ind==1:
		dV = dV_0_phi(f1,f2) + dV_int(f1,f2,1)
	if ind==2:
		dV = dV_0_chi(f1,f2) + dV_int(f1,f2,2)
	if mpl_n==True:
		dV = dV/mpl[0]**2
	return dV
