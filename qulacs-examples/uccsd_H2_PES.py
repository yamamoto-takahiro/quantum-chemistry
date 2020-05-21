import numpy as np

from qulacs import QuantumState
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from qamuy_core.utils.parsers.openfermion_parsers.operator_parser import parse_of_operators
from qamuy_core.algorithms.ansatz import TrotterSingletUCCSD
from qamuy_core.algorithms.eigensolver.vqe import VQE

geometry = [["H", [0, 0, 0]], ["H", [0, 0, 0.700]]]
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
state.set_computational_basis(3)

# ansatz preparation
n_trotter = 1
ansatz = TrotterSingletUCCSD(n_qubits, n_electrons, n_trotter)
# initial parameters preparation
param_size = ansatz.get_parameter_count()
init_params = 2 * np.pi * np.random.random(param_size)

vqe = VQE(qulacs_hamiltonian, ansatz)
options = {"disp": True, "maxiter": 2048, "gtol": 1e-6}
result = vqe.find_minimum([state], "BFGS", init_params, options=options)

results = [result]
geometries = [
    [["H", [0, 0, 0]], ["H", [0, 0, 0.705]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.710]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.715]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.720]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.725]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.730]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.735]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.740]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.745]]],
    [["H", [0, 0, 0]], ["H", [0, 0, 0.750]]],
]
for idx, geometry in enumerate(geometries):
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule)
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    qulacs_hamiltonian = parse_of_operators(n_qubits, jw_hamiltonian)
    init_params = results[idx]["opt_params"]
    vqe = VQE(qulacs_hamiltonian, ansatz)
    result = vqe.find_minimum([state], "BFGS", init_params, options=options)
    results += [result]
print(results)

# [
#     {'opt_params': array([3.14159285, 4.81725634]), 'cost_hist': [[-0.9156662555496518, -1.1074000441166765, -1.1335631153514596, -1.1361833570631608, -1.1361894105280093, -1.136189454065891]], 'nfev': 32, 'nit': 6}, 
#     {'opt_params': array([3.14159275, 4.81822096]), 'cost_hist': [[-1.1364956748548216]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159272, 4.81919268]), 'cost_hist': [[-1.1367503972832005]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159269, 4.82017147]), 'cost_hist': [[-1.1369552252566248]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159269, 4.82115732]), 'cost_hist': [[-1.137111715115465]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159268, 4.82215021]), 'cost_hist': [[-1.137221377072301]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159267, 4.82315029]), 'cost_hist': [[-1.1372856765971089]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159264, 4.82415746]), 'cost_hist': [[-1.137306035753402]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159266, 4.82517177]), 'cost_hist': [[-1.137283834488497]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159263, 4.82619331]), 'cost_hist': [[-1.1372204118806761]], 'nfev': 12, 'nit': 1}, 
#     {'opt_params': array([3.14159262, 4.82722202]), 'cost_hist': [[-1.137117067345729]], 'nfev': 12, 'nit': 1}
# ]