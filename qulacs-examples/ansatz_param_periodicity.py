import numpy as np
from decimal import Decimal

from qulacs import QuantumState
from qulacs.state import inner_product
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from qamuy_core.algorithms.ansatz import TrotterSingletUCCSD, HardwareEfficient
from qamuy_core.utils.parsers.openfermion_parsers.operator_parser import parse_of_operators

mod_2pi = lambda x: float(Decimal(x) % Decimal(2 * np.pi))
vmod_2pi = np.vectorize(mod_2pi)

geometry = [["Li", [0, 0, 0]], ["H", [0, 0, 1.595]]]
basis = "sto-3g"
multiplicity = 1
charge = 0

molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons
fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
qulacs_hamiltonian = parse_of_operators(n_qubits, jw_hamiltonian)

# state preparation
state = QuantumState(n_qubits)
state.set_computational_basis(2 ** n_electrons - 1)

# UCCSD ansatz preparation
n_trotter = 1
ansatz = TrotterSingletUCCSD(n_qubits, n_electrons, n_trotter)
# initial parameters preparation
param_size = ansatz.get_parameter_count()
params = 4 * np.pi * np.random.uniform(0.0, 1.0, param_size)
print(params)
norm_params = vmod_2pi(params)
# norm_params = float(Decimal(params) % Decimal(2*np.pi))
# causes type error:
# TypeError: conversion from numpy.ndarray to Decimal is not supported
print(norm_params)

psi_bra = state.copy()
ansatz.set_all_parameters(params)
ansatz.update_quantum_state(psi_bra)

psi_ket = state.copy()
ansatz.set_all_parameters(norm_params)
ansatz.update_quantum_state(psi_ket)

fid = inner_product(psi_bra, psi_ket)
print(fid)

# HWE ansatz preparation
depth = 10
ansatz = HardwareEfficient(n_qubits, depth)
# initial parameters preparation
param_size = ansatz.get_parameter_count()
params = 4 * np.pi * np.random.uniform(0.0, 1.0, param_size)
print(params)
norm_params = vmod_2pi(params)
print(norm_params)

psi_bra = state.copy()
ansatz.set_all_parameters(params)
ansatz.update_quantum_state(psi_bra)

psi_ket = state.copy()
ansatz.set_all_parameters(norm_params)
ansatz.update_quantum_state(psi_ket)

fid = inner_product(psi_bra, psi_ket)
print(fid)
