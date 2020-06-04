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

geometry = [["Li", [0, 0, 0]], ["H", [0, 0, 1.545]]]
basis = "sto-3g"
multiplicity = 1
charge = 0

# Eigenenergy and eigenstates
molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule)
hamiltonian = molecule.get_molecular_hamiltonian()
sparse_matrix = get_sparse_operator(hamiltonian)
eigs, eigvs = eigsh(sparse_matrix, k=2, which="SA")
print(eigs) # [-7.88276117 -7.80627201]
# print(eigvs[:, 0])
print(np.linalg.norm(eigvs[:, 0]))

# Gradient of energy
dx = 1e-5
deriv = NumericalDerivativeHamiltonian(geometry, basis, multiplicity, charge)
dH_dx_vec = deriv.get_hamiltonian_derivatives(1, h=dx)

for idx_atom in range(len(geometry)):
    for x in range(3):
        dH_dx = get_sparse_operator(dH_dx_vec[idx_atom][x])
        # print(dH_dx.shape)
        if dH_dx.shape == (1, 1):
            continue
        dE_dx = expectation(dH_dx, eigvs[:, 0])
        print(idx_atom, x)
        print(BOHR_TO_ANG * dE_dx)
