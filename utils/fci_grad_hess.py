import operator
from copy import deepcopy
import numpy as np
from scipy.sparse.linalg import eigsh

from openfermion import MolecularData
from openfermion.ops import InteractionOperator
from openfermion.transforms import get_sparse_operator
from openfermionpyscf import run_pyscf


class NumericalDerivativeFCIEnergy:
    def __init__(
        self,
        geometry,
        basis,
        multiplicity,
        charge,
        occupied_indices=None,
        active_indices=None,
    ):
        """
        Args:
            geometry <list>: A list of tuples giving the coordinates of each atom.
                An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
                Distances in angstrom. Use atomic symbols to
                specify atoms.
            basis <string>: A string giving the basis set. An example is 'cc-pvtz'.
            multiplicity <int>: An integer giving the spin multiplicity.
            charge <int>: An integer giving the total molecular charge.
            occupied_indices <list>: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            active_indices <list>: A list of spatial orbital indices indicating
                which orbitals should be considered active.
        """
        self._geometry = geometry
        self._basis = basis
        self._multiplicity = multiplicity
        self._charge = charge
        self._occupied_indices = occupied_indices
        self._active_indices = active_indices

    # public
    def get_fci_energy_derivatives(self, n, h=1e-5, k=1):
        """
        Args:
            n <int>: An order of derivative
            h <float>: A step size
            k <int>: The number of eigenvalues and eigenvectors desired. 
                        k must be smaller than N. It is not possible to compute all eigenvectors of a matrix.

        Returns:
            gradients <numpy.ndarray>: An array with shape of (n_atoms, 3, k) which contains
                gradient of Hamiltonian in the form of openfermion.ops.InteractionOperator
                with respect to the `3*n_atoms` coordinates of atoms. `gradients` is returned when
                `n` is equal to or greater than 1.
            hessians　<numpy.ndarray>: An array with shape of (3*n_atoms, 3*n_atoms, k) which
                contains hessian of Hamiltonian in the form of openfermion.ops.InteractionOperator
                with respect to the `3*n_atoms` coordinates of atoms. `hessians` is returned when
                `n` is equal to or greater than 2.
            third_derivatives <numpy.ndarray>: An array with shape of (3*n_atoms, 3*n_atoms, 3*n_atoms, k)
                which contains 3rd order derivative of Hamiltonian in the form of
                openfermion.ops.InteractionOperator with respect to the `3*n_atoms` coordinates of atoms.
                `third_derivatives` is returned when `n` is equal to 3.
        """
        if n <= 0:
            raise ValueError("Order of derivative, `n`, must be 1 or larger")

        if n > 3:
            raise NotImplementedError("Not implemented yet")
        
        # shape = (n_atoms, 3, 2)
        geometries_grad = self._get_shifted_geometries(self._geometry, h)
        # use of numpy vectorization
        vec_get_fci_energies = np.vectorize(self._get_fci_energies, otypes=[object], signature="(),()->(k)")
        # shape = (n_atoms, 3, 2, k)
        eigs_grad = vec_get_fci_energies(geometries_grad, k)

        # evaluation of gradient
        vec_fci_energy_gradient = np.vectorize(
            self._fci_energy_gradient, otypes=[object], signature="(n,k),()->(k)"
        )
        # shape = (n_atoms, 3, k)
        grads = vec_fci_energy_gradient(eigs_grad, h)
        if n == 1:
            return grads

        # evaluation of diagonal part of hessian
        vec_fci_energy_hessian_diag = np.vectorize(
            self._fci_energy_hessian_diag,
            otypes=[object],
            signature="(k),(n,k),()->(k)",
        )
        # shape = (k,)
        mid_eigs = self._get_fci_energies(self._geometry, k)
        # shape = (n_atoms, 3, k)
        hess_diag = vec_fci_energy_hessian_diag(mid_eigs, eigs_grad, h)
        # evaluation of off-diagonal part of hessian
        # shape = (3*n_atoms, 3*n_atoms, 4)
        geometries_hess = self._get_2d_shifted_geometries(self._geometry, h)
        # shape = (3*n_atoms, 3*n_atoms, 4, k)
        eigs_hess = vec_get_fci_energies(geometries_hess, k)
        vec_fci_energy_hessian_off_diag = np.vectorize(
            self._fci_energy_hessian_off_diag,
            otypes=[object],
            signature="(n,k),()->(k)",
        )
        # shape = (3*n_atoms, 3*n_atoms, k)
        hess_off = vec_fci_energy_hessian_off_diag(eigs_hess, h)
        # merge `hess_diag` to `hess_off`
        n_rows, n_cols, k = hess_diag.shape
        hess_diag = np.reshape(hess_diag, (n_rows * n_cols, k))
        for idx in range(n_rows * n_cols):
            hess_off[idx][idx] = hess_diag[idx]
        if n == 2:
            return grads, hess_off
        else:
            raise ValueError("Invalid choce of `n`")

    # private
    def _get_fci_energies(self, geometry, k):
        """
        Args:
            geometry <list>: e.g. [['H', [0, 0, 0]], ['H', [0, 0, d]]]
            k <int>: The number of eigenvalues and eigenvectors desired.
        Returns:
            fci_energies <list>
        """
        if geometry is None:
            return np.empty((k,), dtype=object)

        molecule = MolecularData(
            geometry,
            basis=self._basis,
            multiplicity=self._multiplicity,
            charge=self._charge,
        )
        molecule = run_pyscf(molecule, run_scf=True)
        hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=self._occupied_indices, active_indices=self._active_indices
        )
        sparse_matrix = get_sparse_operator(hamiltonian)
        eigs, _ = eigsh(sparse_matrix, k=k, which="SA")
        return eigs


    def _fci_energy_gradient(self, eigs, h):
        """
        Args:
            eigs <list>: A list of core energies of [eig(a-h), eig(a+h)]
            h <float>: step size

        Returns:
            first derivative evaluated as
                (f(a+h) - f(a-h))/(2*h)
        """
        return (eigs[1] - eigs[0]) / (2 * h)

    def _fci_energy_hessian_diag(self, mid_eigs, eigs, h):
        """
        Args:
            mid_eigs <float>: eig(a)
            eigs <list>: A list of eigenvalues, [eig(a-h), eig(a+h)]
            h <float>: step size

        Retunrs:
            diagonal part of second derivative evaluated as
                (f(a-h) + f(a+h) - 2 * f(a))/(h*h)
        """
        return (eigs[0] - 2.0 * mid_eigs + eigs[1]) / (h * h)

    def _fci_energy_hessian_off_diag(self, eigs, h):
        """
        Args:
            eigs <list>: A list of eigenvalues, [eig(a-h1, b-h2), eig(a-h1, b+h2), eig(a+h1, b-h2), eig(a+h1, b+h2)]
            h <float>: step size

        Returns:
            off-diagonal part of second derivative evaluated as
                (f(a-h1, b-h2) - f(a-h1, b+h2) - f(a+h1, b-h2) + f(a+h1, b+h2))/(4*h*h)
        """
        if np.isnan(eigs.astype(float)).any():
            n_atoms, k = eigs.shape
            return np.empty((k,), dtype=object)
        denom = 4 * h * h
        return (eigs[0] - eigs[1] - eigs[2] + eigs[3]) / denom

    def _fci_energy_third_derivative(
        self, constants, one_body_tensors, two_body_tensors, h
    ):
        # TODO:
        raise NotImplementedError("Not implemented yet")

    def _get_shifted_geometries(self, geometry, h):
        n_atoms = len(geometry)
        shifted_geometries = np.empty((n_atoms, 3, 2), dtype=object)
        for idx_atom, shifted_atom in enumerate(geometry):
            new_shifted_atom = self._shift_atom(shifted_atom, h, ndim=1)
            for idx_pos in range(3):
                for idx_sign in range(2):
                    new_geometry = deepcopy(geometry)
                    new_geometry[idx_atom] = new_shifted_atom[idx_pos][idx_sign]
                    shifted_geometries[idx_atom][idx_pos][idx_sign] = new_geometry
        return shifted_geometries

    def _get_2d_shifted_geometries(self, geometry, h):
        n_atoms = len(geometry)
        shifted_geometries = np.empty((3 * n_atoms, 3 * n_atoms, 4), dtype=object)
        k_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for xi in range(3 * n_atoms):
            idx_atom1 = xi // 3
            x_atom1 = xi % 3
            for xj in range(3 * n_atoms):
                if xj <= xi:
                    shifted_geometries[xi][xj] = np.array([None for _ in range(4)])
                    continue
                idx_atom2 = xj // 3
                x_atom2 = xj % 3
                if idx_atom1 == idx_atom2:
                    shifted_atom = geometry[idx_atom1]
                    # shape = (3, 4)
                    new_shifted_atom = self._shift_atom(shifted_atom, h, ndim=2)
                    for k in range(4):
                        new_geometry = deepcopy(geometry)
                        new_geometry[idx_atom1] = new_shifted_atom[
                            x_atom1 + x_atom2 - 1
                        ][k]
                        shifted_geometries[xi][xj][k] = new_geometry
                else:
                    shifted_atom1 = geometry[idx_atom1]
                    shifted_atom2 = geometry[idx_atom2]
                    # shape = (3, 2)
                    new_shifted_atom1 = self._shift_atom(shifted_atom1, h, ndim=1)
                    # shape = (3, 2)
                    new_shifted_atom2 = self._shift_atom(shifted_atom2, h, ndim=1)
                    for k in range(4):
                        new_geometry = deepcopy(geometry)
                        k1, k2 = k_indices[k]
                        new_geometry[idx_atom1] = new_shifted_atom1[x_atom1][k1]
                        new_geometry[idx_atom2] = new_shifted_atom2[x_atom2][k2]
                        shifted_geometries[xi][xj][k] = new_geometry
        return shifted_geometries

    def _shift_atom(self, shifted_atom, h, ndim=1):
        symbol = shifted_atom[0]
        coordinate = shifted_atom[1]
        if ndim == 1:
            new_shifted_atom = [
                [
                    [symbol, self._add_coordinates(coordinate, [-h, 0, 0])],
                    [symbol, self._add_coordinates(coordinate, [h, 0, 0])],
                ],
                [
                    [symbol, self._add_coordinates(coordinate, [0, -h, 0])],
                    [symbol, self._add_coordinates(coordinate, [0, h, 0])],
                ],
                [
                    [symbol, self._add_coordinates(coordinate, [0, 0, -h])],
                    [symbol, self._add_coordinates(coordinate, [0, 0, h])],
                ],
            ]
        elif ndim == 2:
            new_shifted_atom = [
                [
                    [symbol, self._add_coordinates(coordinate, [-h, -h, 0])],
                    [symbol, self._add_coordinates(coordinate, [-h, h, 0])],
                    [symbol, self._add_coordinates(coordinate, [h, -h, 0])],
                    [symbol, self._add_coordinates(coordinate, [h, h, 0])],
                ],
                [
                    [symbol, self._add_coordinates(coordinate, [-h, 0, -h])],
                    [symbol, self._add_coordinates(coordinate, [-h, 0, h])],
                    [symbol, self._add_coordinates(coordinate, [h, 0, -h])],
                    [symbol, self._add_coordinates(coordinate, [h, 0, h])],
                ],
                [
                    [symbol, self._add_coordinates(coordinate, [0, -h, -h])],
                    [symbol, self._add_coordinates(coordinate, [0, -h, h])],
                    [symbol, self._add_coordinates(coordinate, [0, h, -h])],
                    [symbol, self._add_coordinates(coordinate, [0, h, h])],
                ],
            ]
        else:
            raise NotImplementedError("Not implemented in case where `ndim > 2`")
        return new_shifted_atom

    def _add_coordinates(self, c1, c2):
        return list(map(operator.add, c1, c2))

if __name__ == "__main__":
    geometry = [['H', [0, 0, 0]], ['H', [0, 0, 0.7414]]]
    # geometry = [['H', [0, 0, 0]], ['H', [0, 0, 0.75]]]
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    deriv =  NumericalDerivativeFCIEnergy(geometry, basis, multiplicity, charge)
    grad, hess = deriv.get_fci_energy_derivatives(2, k=2)
    print("grad")
    print(grad)
    print("hess")
    print(hess)
