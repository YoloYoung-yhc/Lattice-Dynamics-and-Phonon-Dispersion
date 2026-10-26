"""Evaluate normal modes on Cartesian wavevector paths.

Branches are sorted by frequency independently at each wavevector. Mode
tracking across crossings and degenerate subspaces is not performed.
"""

from dataclasses import dataclass

import numpy as np

from dynamical_matrix import (
    DynamicalMatrixBuilder,
    make_diatomic_chain,
    make_monatomic_chain,
    real_array,
    solve_modes,
    validate_model,
)


@dataclass
class DispersionResult:
    """Store a wavevector path and its independent normal-mode solutions.

    Attributes:
        q_points (np.ndarray): Cartesian wavevectors, shape (Nq, dimension).
        eigenvalues (np.ndarray): Original squared frequencies, shape (Nq, num_dof).
        frequencies (np.ndarray): Angular frequencies, shape (Nq, num_dof).
        eigenvectors (np.ndarray): Column-wise modes, shape (Nq, num_dof, num_dof).
        clipped_negative (np.ndarray): Small-negative conversion flags.
        diagnostics (dict): One diagnostic array per numerical residual or scale.
        model_checks (dict): Force and mass-weighted translation checks.
    """

    q_points: np.ndarray
    eigenvalues: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    clipped_negative: np.ndarray
    diagnostics: dict
    model_checks: dict


def calculate_dispersion(lattice_model, wavevectors, *, atol=1e-12, rtol=1e-10):
    """Evaluate normal modes on a Cartesian wavevector array.

    Eigenvectors are columns. Branch ordering is local to each wavevector;
    compare degenerate subspaces instead of individual vectors when modes
    are degenerate.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        wavevectors (array_like): Cartesian wavevectors of shape (Nq,
            dimension), in inverse a0 units.
        atol (float): Nonnegative absolute tolerance in the units of the
            checked quantity.
        rtol (float): Nonnegative dimensionless relative tolerance.

    Returns:
        DispersionResult: Eigenvalues and frequencies of shape (Nq,
            num_dof), eigenvectors of shape (Nq, num_dof, num_dof), and
            diagnostics.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    wavevectors = real_array(wavevectors, "wavevectors")
    if (
        wavevectors.ndim != 2
        or wavevectors.shape[0] == 0
        or wavevectors.shape[1] != lattice_model.dimension
    ):
        raise ValueError(
            f"wavevectors must have nonempty shape (Nq,{lattice_model.dimension}) array"
        )
    model_checks = validate_model(lattice_model)
    # Geometry, force checks, and mass factors are shared across the path.
    matrix_builder = DynamicalMatrixBuilder(lattice_model)
    eigenvalue_scale = matrix_builder.eigenvalue_scale
    frequency_shape = (len(wavevectors), lattice_model.n_dof)
    eigenvalues = np.empty(frequency_shape)
    frequencies = np.empty(frequency_shape)
    eigenvectors = np.empty(
        (*frequency_shape, lattice_model.n_dof), dtype=np.complex128
    )
    clipped_negative = np.empty(frequency_shape, dtype=bool)
    diagnostics = {}
    for wavevector_index, wavevector in enumerate(wavevectors):
        try:
            dynamical_matrix = matrix_builder.build(wavevector)
            mode_result = solve_modes(
                dynamical_matrix,
                eigenvalue_scale=eigenvalue_scale,
                atol=atol,
                rtol=rtol,
            )
        except ValueError as error:
            raise ValueError(
                f"q[{wavevector_index}]={wavevector}: {error}"
            ) from error
        eigenvalues[wavevector_index] = mode_result.eigenvalues
        frequencies[wavevector_index] = mode_result.frequencies
        eigenvectors[wavevector_index] = mode_result.eigenvectors
        clipped_negative[wavevector_index] = mode_result.clipped_negative
        for key, value in mode_result.diagnostics.items():
            if key not in diagnostics:
                diagnostics[key] = np.empty(len(wavevectors))
            diagnostics[key][wavevector_index] = value
    return DispersionResult(
        wavevectors,
        eigenvalues,
        frequencies,
        eigenvectors,
        clipped_negative,
        diagnostics,
        model_checks,
    )


def main():
    """Print dispersion-array shapes and numerical residual summaries.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    for model_name, lattice_model in [
        ("Monatomic chain", make_monatomic_chain()),
        ("Diatomic chain", make_diatomic_chain()),
    ]:
        lattice_constant = lattice_model.lattice_vectors[0, 0]
        wavevectors = np.linspace(
            -np.pi / lattice_constant, np.pi / lattice_constant, 401
        )
        wavevectors[200] = 0.0
        dispersion_result = calculate_dispersion(
            lattice_model, wavevectors[:, None]
        )
        print(
            f"\n{model_name}: {len(wavevectors)} wavevectors, {lattice_model.n_dof} branches"
        )
        print("Frequency array shape: ", dispersion_result.frequencies.shape)
        print("Eigenvector array shape: ", dispersion_result.eigenvectors.shape)
        print("Raw Gamma eigenvalues: ", dispersion_result.eigenvalues[200])
        print("Boundary eigenvalues: ", dispersion_result.eigenvalues[-1])
        print(
            "Maximum scaled eigen-equation residual: ",
            np.max(
                dispersion_result.diagnostics["scaled_eigen_equation_residual"]
            ),
        )
        print(
            "Maximum orthogonality residual: ",
            np.max(dispersion_result.diagnostics["orthogonality_residual"]),
        )


if __name__ == "__main__":
    main()
