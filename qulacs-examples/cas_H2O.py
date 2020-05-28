import numpy as np
from scipy.sparse.linalg import eigsh

from qulacs import QuantumState
from openfermion.hamiltonians import MolecularData
from openfermion.ops import InteractionOperator
from openfermion.transforms import get_sparse_operator, get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from qamuy_core.utils.parsers.openfermion_parsers.operator_parser import parse_of_operators
from qamuy_core.algorithms.ansatz import SymmetryPreservingReal
from qamuy_core.algorithms.eigensolver.vqe import VQE


d = 0.9584
a = np.radians(104.45)
# geometry = [["H", [d, 0, 0]], ["O", [0, 0, 0]],["H", [d*np.cos(a), d*np.sin(a), 0]]]
geometry = [["H", [0, 0, 0]], ["O", [0, 1.0, 0]], ["H", [0, 0, 1.0]]]
basis = "sto-3g"
multiplicity = 1
charge = 0

molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule)

# CAS(2, 2) 
# n_active_eles = 2
# n_active_orbs = 2
# CAS(6, 8)
n_active_eles = 8
n_active_orbs = 6
n_qubits = 2 * n_active_orbs
n_core_orbs = (molecule.n_electrons - n_active_eles) // 2
occupied_indices = list(range(n_core_orbs))
active_indices = [n_core_orbs + i for i in range(n_active_orbs)]
print(occupied_indices) # [0, 1, 2, 3]
print(active_indices) # [4, 5]
core_constant, one_body_integrals, two_body_integrals = molecule.get_active_space_integrals(occupied_indices, active_indices)
# print(core_constant) # -81.48580476764542

active_space_hamiltonian = InteractionOperator(core_constant, one_body_integrals, two_body_integrals)


sparse_matrix = get_sparse_operator(active_space_hamiltonian)
eigs, _ = eigsh(sparse_matrix, k=2, which="SA")
print(eigs) # [-82.75752396 -82.35498755]

active_space_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices, active_indices)
sparse_matrix = get_sparse_operator(active_space_hamiltonian)
eigs, _ = eigsh(sparse_matrix, k=2, which="SA")
print(eigs) # [-74.92541935 -74.67780361]

fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian(occupied_indices, active_indices))
jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
qlx_hamiltonian = parse_of_operators(n_qubits, jw_hamiltonian)

# state preparation
state = QuantumState(n_qubits)
state.set_computational_basis(3)

# ansatz preparation
depth = 4
ansatz = SymmetryPreservingReal(n_qubits, depth)

# initial parameters preparation
param_size = ansatz.get_parameter_count()
init_params = 2 * np.pi * np.random.random(param_size)
# 
vqe = VQE(qlx_hamiltonian, ansatz)
options = {"disp": True, "maxiter": 2048, "gtol": 1e-6}
result = vqe.find_minimum([state], "BFGS", init_params, options=options)

print(result)
# {
#     'opt_params': array([4.05815707, 6.21797, 2.51155268, 5.55120094, 1.48535056, -1.51046141, 3.8419363, 0.17185733, 3.65803586, 2.19582281, 3.07460883, 2.09022427]), 
#     'cost_hist': [[-74.58361709507535, -74.69993412807506, -74.73101447659766, -74.77368990213131, -74.82354105284362, -74.83956014882254, -74.86968790968402, -74.92681137233532, -74.95882582235167, -74.96398969635251, -74.96407769492207, -74.9642100451351, -74.96435556469851, -74.96436630512463, -74.96436830978475, -74.96436832567865, -74.96436832615382]], 
#     'nfev': 572, 
#     'nit': 17
# }
