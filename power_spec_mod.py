# power_spec_mod.py

# Module for computing the dimensionless power spectrum.

# to do: write a function that takes P_{xx} as input and outputs \mathcal{P}_x as output

import numpy as np

# Function to calculate power from a given DFT correlation
# spec_dft: \langle \tilde{f}^{(D)}_n\tilde{f}^{(D)}_m \rangle on a radial profile
# L: side length of lattice
# N: number of sites on lattice size
def pspec_3d(spec_dft, L, N):
    dx = L/N
    dk = 2.*np.pi/L
    n_max = np.shape(spec_dft)[-1]  # number of entries in spec_dft
    n = np.arange(0,n_max+1)
    P = (2.*np.pi**2)*dk**3*dx**3 * spec_dft * n**3/N**3
    return P

# Function to calculate spectrum from a given DFT correlation
# spec_dft: \langle \tilde{f}^{(D)}_n\tilde{f}^{(D)}_m \rangle on a radial profile
# L: side length of lattice
# N: number of sites on lattice size
def spec_3d(spec_dft, L, N):
    dx = L/N
    P = dx**3/N**3 * spec_dft
    return P
