import numpy as np

from pyscf import gto, scf, mcscf
from pyscf import grad

print("H2O/CAS(2,2)/STO-3G")
d = 0.9584
a = np.radians(104.45)
geometry = [
    ["H", [d, 0, 0]],
    ["O", [0, 0, 0]],
    ["H", [d * np.cos(a), d * np.sin(a), 0]],
]
print(geometry)
# occupied_indices = [0, 1, 2, 3]
# active_indices = [4, 5]
mol = gto.Mole()
mol.atom = geometry
mol.charge = 0
mol.spin = 0
mol.basis = "sto-3g"
mol.symmetry = False
mol.build()
mf = scf.RHF(mol).run()
# 2 orbitals, 2 electrons
mc = mcscf.CASCI(mf, 2, 2).run()
g = mc.Gradients()
g.run()

print("H2O/CAS(6,8)/STO-3G")
occupied_indices = [0]
active_indices = [1, 2, 3, 4, 5, 6]
d = 0.9584
a = np.radians(104.45)
geometry = [
    ["H", [d, 0, 0]],
    ["O", [0, 0, 0]],
    ["H", [d * np.cos(a), d * np.sin(a), 0]],
]
print(geometry)
# occupied_indices = [0]
# active_indices = [1, 2, 3, 4, 5, 6]
mol = gto.Mole()
mol.atom = geometry
mol.charge = 0
mol.spin = 0
mol.basis = "sto-3g"
mol.symmetry = False
mol.build()
mf = scf.RHF(mol).run()
# 6 orbitals, 8 electrons
mc = mcscf.CASCI(mf, 6, 8).run()
g = mc.Gradients()
g.run()

# Ref: https://sunqm.github.io/pyscf/mcscf.html