# cumulant_mod.py

import numpy as np

# use dictionary mapping for this
def cumulant(f, axis, n):
    case = {
        2: moment(f, axis, 2),
        3: moment(f, axis, 3),
        4: moment(f, axis, 4) - 3.*moment(f, axis, 2),
        5: moment(f, axis, 5) - 10.*moment(f, axis, 3)*moment(f, axis, 2),
        6: moment(f, axis, 6) - 15.*moment(f, axis, 4)*moment(f, axis, 2) \
        - 10.*moment(f, axis, 3)**2 +30.*moment(f, axis, 2)**3
        }
    kappa = case.get(n)
    return kappa

# to do add dof to this calc
def moment(f, axis, n):
    mu_0 = np.mean(f)
    d = 1
    for a in axis:
        d = d*np.shape(f)[a]
    mu_n = np.sum((f-mu_0)**n)/(d-1)
    return mu_n
