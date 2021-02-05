# potential.py

# Module containing functions for evaluating the potential and its derivatives

# to do: extent dV_int to other potential options

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

POTOPT = np.ones(1)                     # interaction potential option
### Input functions

# Initialize mpl
def init_mpl(mpl_in):
	mpl[0] = mpl_in
	return

# Initialize potential parameters
def init_param(phi_p_in, phi_w_in, m2_p_in, m2_phi_in=1., m2_inf_in=1., lambda_chi_in=10., POTOPT_in=2):
	phi_p[0] = phi_p_in
	phi_w[0] = phi_w_in
	m2_p[0] = m2_p_in
	m2_phi[0] = m2_phi_in
	m2_inf[0] = m2_inf_in
	lambda_chi[0] = lambda_chi_in
	POTOPT[0] = POTOPT_in
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

def V_int(f1, f2, mpl_in=False):
        if (POTOPT[0]==2):
                V_int = V_int_opt2(f1, f2, mpl_in)
        elif (POTOPT[0]==4):
                V_int = V_int_opt4(f1, f2, mpl_in)
        return V_int

# phi-chi interaction potential for POTOPT 2
def V_int_opt2(f1, f2, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	V_int = 0.5*(-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0]))*( m2_p[0]-m2_inf[0] + g2*(f1-phi_p[0])**2 - g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4) * f2**2
	if mpl_n==True:
		V_int = V_int/mpl[0]**2
	return V_int

# phi-chi interaction potential for POTOPT 4
def V_int_opt4(f1, f2, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	V_int = 0.5*(0.5*np.sign(phi_p[0]+phi_w[0]-f1)-0.5*np.sign(phi_p-f1)) \
                *(m2_p[0]-m2_inf[0]+g2*(f1-phi_p[0])**2-g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4)*f2**2 \
                +0.5*(0.5*np.sign(phi_p[0]-f1)+0.5) \
                *(m2_p[0] - m2_inf[0])*f2**2
	if mpl_n==True:
		V_int = V_int/mpl[0]**2
	return V_int

# chi effective mass term (2nd chi partial derivative at chi=0)
def m2eff_chi(f1, mpl_n=False):
        if (POTOPT[0]==2):
                m2eff = m2eff_chi_opt2(f1, mpl_n)
        elif (POTOPT[0]==4):
                m2eff = m2eff_chi_opt4(f1, mpl_n)
        return m2eff

# chi effective mass term (2nd chi partial derivative at chi=0) for POTOPT 2
def m2eff_chi_opt2(f1, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	m2eff = (-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0]))*( m2_p[0]-m2_inf[0] + g2*(f1-phi_p[0])**2 - g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4)+m2_inf[0]
	if mpl_n==True:
		m2eff = m2eff/mpl[0]**2
	return m2eff

# chi effective mass term (2nd chi partial derivative at chi=0) for POTOPT 4
def m2eff_chi_opt4(f1, mpl_n=False):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	m2eff = (0.5*np.sign(phi_p[0]+phi_w[0]-f1)-0.5*np.sign(phi_p-f1)) \
                *(m2_p[0]-m2_inf[0]+g2*(f1-phi_p[0])**2-g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4) \
                +(0.5*np.sign(phi_p[0]-f1)+0.5) \
                *(m2_p[0] - m2_inf[0])
	if mpl_n==True:
		m2eff = m2eff/mpl[0]**2
	return m2eff

# chi at the minimum of the trough
def chi_min(f1):
	g2 = 2.*(m2_inf[0]-m2_p[0])/(phi_w[0]**2)
	chi_min = np.where((m2eff_chi(f1)>0),0.,np.sqrt(-0.5*m2eff_chi(f1)/(0.5*lambda_chi[0])))
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
        V_0 = 0.5*m2_phi[0]*f1**2 + 0.5*m2_inf[0]*f2**2
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

# Second derivatives
def ddV(f1, f2, ind1, ind2, mpl_n=False):
        if (ind1==1 and ind2==1):
                ddV = ddV_f1f1(f1,f2)
        elif (ind1==2 and ind2==2):
                ddV = ddV_f2f2(f1,f2)
        else:
                ddV = ddV_f1f2(f1,f2)
        if mpl_n:
                ddV = ddV/mpl[0]**2
        return ddV

def ddV_f1f1(f1,f2):
        if(POTOPT[0]==2):
                ddV = ddV_f1f1_opt2(f1,f2)
        return ddV

def ddV_f1f2(f1,f2):
        if(POTOPT[0]==2):
                ddV = ddV_f1f2_opt2(f1,f2)
        return ddV

def ddV_f2f2(f1,f2):
        if(POTOPT[0]==2):
                ddV = ddV_f2f2_opt2(f1,f2)
        return ddV

def ddV_f1f1_opt2(f1,f2):
        ddV = m2_phi \
                +0.5*(-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0])) \
                *(2.*g2 - 3.*g2**2/((m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**2) * f2**2
        return ddV

def ddV_f1f2_opt2(f1,f2):
        ddV = (-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0])) \
                *(2.*g2*(f1-phi_p[0]) - g2**2/((m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**3) * f2
        return ddV

def ddV_f2f2_opt2(f1,f2):
        ddV = m2_inf + 3.*lambda_chi*f2**2 \
                +(-0.5*np.sign(-f1+phi_p[0]-phi_w[0])-0.5*np.sign(f1-phi_p[0]-phi_w[0])) \
                *( m2_p[0]-m2_inf[0] + g2*(f1-phi_p[0])**2 - g2**2/(4.*(m2_inf[0]-m2_p[0]))*(f1-phi_p[0])**4)
        return ddV
