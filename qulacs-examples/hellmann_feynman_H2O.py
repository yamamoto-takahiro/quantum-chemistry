import operator
from copy import deepcopy
import numpy as np
from scipy.sparse.linalg import eigsh

from openfermion import MolecularData
from openfermion.ops import InteractionOperator
from openfermion.transforms import get_sparse_operator
from openfermion.utils import expectation
from openfermionpyscf import run_pyscf
from pyscf import lib
from qamuy_core.chemistry import NumericalDerivativeHamiltonian


BOHR_TO_ANG = lib.param.BOHR

# H2O with CAS(2,2)
d = 0.9584
a = np.radians(104.45)
geometry = [
    ["H", [d, 0, 0]],
    ["O", [0, 0, 0]],
    ["H", [d * np.cos(a), d * np.sin(a), 0]],
]
# occupied_indices = None
# active_indices = None
# CAS(2,2)
# occupied_indices = [0, 1, 2, 3]
# active_indices = [4, 5]
# CAS(6,8)
occupied_indices = [0]
active_indices = [1, 2, 3, 4, 5, 6]
basis = "sto-3g"
multiplicity = 1
charge = 0

# Eigenenergy and eigenstates
molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule)
active_space_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices, active_indices)
sparse_matrix = get_sparse_operator(active_space_hamiltonian)
eigs, eigvs = eigsh(sparse_matrix, k=2, which="SA")
print(eigs) # [-74.96436833 -74.57190201]
# print(eigvs[:, 0])
print(np.linalg.norm(eigvs[:, 0]))

# Gradient of energy
dx = 1e-5
deriv = NumericalDerivativeHamiltonian(geometry, basis, multiplicity, charge, occupied_indices=occupied_indices, active_indices=active_indices)
dH_dx_vec = deriv.get_hamiltonian_derivatives(1, h=dx)

for idx_atom in range(len(geometry)):
    for x in range(3):
        dH_dx = get_sparse_operator(dH_dx_vec[idx_atom][x])
        # print(dH_dx.shape)
        if dH_dx.shape == (1, 1):
            continue
        print(idx_atom, x)
        dE_dx = expectation(dH_dx, eigvs[:, 0])
        print(BOHR_TO_ANG * dE_dx)