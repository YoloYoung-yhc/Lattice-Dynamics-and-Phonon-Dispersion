"""Define harmonic lattice models and solve mass-normalized normal modes.

Real-space lattice vectors are rows. The cell-phase convention uses only
cell translations in Bloch factors. Spatial and displacement dimensions
coincide; one-dimensional and in-plane two-dimensional models are supported.
"""

from dataclasses import dataclass

import numpy as np


def real_array(values, parameter_name):
    """Return a finite real-valued float64 array.

    Args:
        values (array_like): Input values; complex and nonfinite values are
            rejected.
        parameter_name (str): Parameter label used in validation errors.

    Returns:
        np.ndarray: A float64 copy with the same shape as the input.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError(
            f"{parameter_name} must contain only finite real values"
        )
    return values.astype(np.float64)


def nonnegative_scalar(value, parameter_name):
    """Validate and return a nonnegative finite real scalar.

    Args:
        value (int or float): Scalar to validate.
        parameter_name (str): Parameter label used in validation errors.

    Returns:
        float: The validated scalar.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    value = real_array(value, parameter_name)
    if value.shape != () or value < 0:
        raise ValueError(
            f"{parameter_name} must be a nonnegative finite real scalar"
        )
    return float(value)


def scaled_tolerance(scale, atol, rtol):
    """Combine absolute and relative tolerances with consistent units.

    Args:
        scale (float): Nonnegative reference scale in the same units as
            atol.
        atol (float): Nonnegative absolute tolerance in the units of the
            checked quantity.
        rtol (float): Nonnegative dimensionless relative tolerance.

    Returns:
        float: Absolute tolerance plus relative tolerance times scale.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    scale = nonnegative_scalar(scale, "scale")
    atol = nonnegative_scalar(atol, "atol")
    rtol = nonnegative_scalar(rtol, "rtol")
    tolerance = atol + rtol * scale
    if not np.isfinite(tolerance):
        raise ValueError("Tolerance overflow; check parameter scales")
    return tolerance


@dataclass
class LatticeModel:
    """Store harmonic lattice data in the primitive-cell phase convention.

    Attributes:
        lattice_vectors (np.ndarray): Shape (dimension, dimension), with one
            real-space basis vector per row, in a0 units.
        basis_positions (np.ndarray): Shape (num_atoms, dimension), in a0 units.
        masses (np.ndarray): Shape (num_atoms,), with positive masses in m0 units.
        force_constants (dict): Integer cell translations mapped to real
            (num_dof, num_dof) force blocks. Rows refer to the reference cell;
            columns refer to the translated cell. Ordering is atom-major,
            followed by Cartesian components within each atom.
    """

    lattice_vectors: np.ndarray
    basis_positions: np.ndarray
    masses: np.ndarray
    force_constants: dict[tuple[int, ...], np.ndarray]

    def __post_init__(self):
        """Normalize model arrays and validate their shapes and constraints.

        Raises:
            ValueError: If inputs violate the documented domain or physical
                constraints.
        """
        self.lattice_vectors = real_array(
            self.lattice_vectors, "lattice_vectors"
        )
        if self.lattice_vectors.shape not in [(1, 1), (2, 2)]:
            raise ValueError(
                "lattice_vectors must have shape (1, 1) or (2, 2), with one basis vector per row"
            )
        if np.linalg.matrix_rank(self.lattice_vectors) != self.dimension:
            raise ValueError(
                "Lattice basis vectors must be linearly independent"
            )
        self.masses = real_array(self.masses, "masses")
        if (
            self.masses.ndim != 1
            or self.masses.size == 0
            or np.any(self.masses <= 0)
        ):
            raise ValueError(
                "masses must be a nonempty one-dimensional positive array"
            )
        self.basis_positions = real_array(
            self.basis_positions, "basis_positions"
        )
        if self.basis_positions.shape != (self.n_atoms, self.dimension):
            raise ValueError(
                "basis_positions must have shape (num_atoms, dimension)"
            )
        if (
            not isinstance(self.force_constants, dict)
            or not self.force_constants
        ):
            raise ValueError("force_constants must be a nonempty dictionary")
        force_constant_blocks = {}
        for cell_shift, force_constant_block in self.force_constants.items():
            if (
                not isinstance(cell_shift, tuple)
                or len(cell_shift) != self.dimension
                or any(
                    (
                        not isinstance(translation_index, (int, np.integer))
                        or isinstance(translation_index, (bool, np.bool_))
                        for translation_index in cell_shift
                    )
                )
            ):
                raise ValueError(
                    "Force-constant keys must be integer translation tuples matching the spatial dimension"
                )
            force_constant_block = real_array(
                force_constant_block, f"force_constants[{cell_shift}]"
            )
            if force_constant_block.shape != (self.n_dof, self.n_dof):
                raise ValueError(
                    f"Translation {cell_shift} force-constant block must have shape ({self.n_dof},{self.n_dof})"
                )
            force_constant_blocks[cell_shift] = force_constant_block
        self.force_constants = force_constant_blocks
        validate_force_constants(self)

    @property
    def dimension(self):
        """Return the shared spatial and displacement dimension.

        Returns:
            int: Spatial dimension, either one or two.
        """
        return self.lattice_vectors.shape[0]

    @property
    def n_atoms(self):
        """Return the number of atoms in the primitive cell.

        Returns:
            int: Number of atoms in one primitive cell.
        """
        return self.masses.size

    @property
    def n_dof(self):
        """Return the number of displacement degrees of freedom per cell.

        Returns:
            int: Number of displacement degrees of freedom per primitive cell.
        """
        return self.n_atoms * self.dimension

    @property
    def dof_masses(self):
        """Return masses in atom-major displacement order.

        Returns:
            np.ndarray: Shape (num_dof,), with each atomic mass repeated per
                direction.
        """
        return np.repeat(self.masses, self.dimension)

    @property
    def eigenvalue_scale(self):
        """Return a wavevector-independent bound on dynamical-matrix norms.

        The scale is the sum of the spectral norms of mass-normalized real-space
        force blocks, providing a common bound for all wavevectors.

        Returns:
            float: Sum of mass-normalized force-block spectral norms, in omega0
                squared units.
        """
        square_root_masses = np.sqrt(self.dof_masses)
        return float(
            sum(
                (
                    np.linalg.norm(
                        force_constant_block
                        / square_root_masses[:, None]
                        / square_root_masses[None, :],
                        ord=2,
                    )
                    for force_constant_block in self.force_constants.values()
                )
            )
        )


def validate_force_constants(lattice_model, *, atol=1e-12, rtol=1e-10):
    """Check force-constant reciprocity and the acoustic sum rule.

    The absolute tolerance has force-constant units. Each Cartesian
    translation direction is checked separately; summing all matrix columns
    together would be insufficient in two dimensions.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        atol (float): Nonnegative absolute tolerance in the units of the
            checked quantity.
        rtol (float): Nonnegative dimensionless relative tolerance.

    Returns:
        dict[str, float]: Force-unit residuals, scale, and acceptance
            tolerance.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    force_constant_scale = float(
        sum(
            (
                np.linalg.norm(force_constant_block, ord=2)
                for force_constant_block in lattice_model.force_constants.values()
            )
        )
    )
    tolerance = scaled_tolerance(force_constant_scale, atol, rtol)
    reciprocity_residual = 0.0
    total_force_constants = np.zeros((lattice_model.n_dof, lattice_model.n_dof))
    for (
        cell_shift,
        force_constant_block,
    ) in lattice_model.force_constants.items():
        opposite = tuple(
            (-translation_index for translation_index in cell_shift)
        )
        reciprocal_block = lattice_model.force_constants.get(
            opposite, np.zeros_like(force_constant_block)
        )
        reciprocity_residual = max(
            reciprocity_residual,
            float(np.max(np.abs(force_constant_block - reciprocal_block.T))),
        )
        total_force_constants += force_constant_block
    if reciprocity_residual > tolerance:
        raise ValueError(
            f"Force-constant reciprocity failed: residual={reciprocity_residual}"
        )
    # Test each Cartesian translation separately, including shear directions.
    translation_vectors = np.tile(
        np.eye(lattice_model.dimension), (lattice_model.n_atoms, 1)
    )
    sum_rule_residual = float(
        np.max(np.abs(total_force_constants @ translation_vectors))
    )
    if sum_rule_residual > tolerance:
        raise ValueError(
            f"Acoustic sum rule failed: residual={sum_rule_residual}"
        )
    return {
        "force_constant_scale": force_constant_scale,
        "force_constant_tolerance": tolerance,
        "reciprocity_residual": reciprocity_residual,
        "acoustic_sum_rule_residual": sum_rule_residual,
    }


def build_dynamical_matrix(lattice_model, wavevector):
    """Assemble a mass-normalized dynamical matrix at one wavevector.

    The matrix follows the primitive-cell phase convention. Its eigenvalues
    have squared-angular-frequency units; actual displacements are recovered
    from mass-weighted eigenvectors by dividing each atomic component by its
    square-root mass.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        wavevector (array_like): Cartesian wavevector of shape (dimension,),
            in inverse a0 units.

    Returns:
        np.ndarray: A complex128 matrix whose eigenvalues are squared
            angular frequencies.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    return DynamicalMatrixBuilder(lattice_model).build(wavevector)


class DynamicalMatrixBuilder:
    """Prepare model-dependent quantities once for a wavevector batch.

    The prepared data are local to one calculation. A later calculation
    rebuilds them so edits to a model cannot leave a stale global cache.
    """

    def __init__(self, lattice_model):
        """Validate the model and snapshot its Fourier assembly data.

        Args:
            lattice_model (LatticeModel): Model to prepare for assembly.

        Raises:
            ValueError: If force reciprocity or the acoustic sum rule fails.
        """
        validate_force_constants(lattice_model)
        self.dimension = lattice_model.dimension
        self.num_dof = lattice_model.n_dof
        self.square_root_masses = np.sqrt(lattice_model.dof_masses)
        self.eigenvalue_scale = lattice_model.eigenvalue_scale
        self.hermitian_tolerance = scaled_tolerance(
            self.eigenvalue_scale, 1e-12, 1e-10
        )
        self.force_blocks = tuple(
            (
                np.asarray(cell_shift) @ lattice_model.lattice_vectors,
                force_constant_block.copy(),
            )
            for cell_shift, force_constant_block in lattice_model.force_constants.items()
        )

    def build(self, wavevector):
        """Assemble and validate a dynamical matrix using prepared data.

        Args:
            wavevector (array_like): Cartesian wavevector, shape (dimension,).

        Returns:
            np.ndarray: Complex matrix in squared angular-frequency units.

        Raises:
            ValueError: If the wavevector or assembled matrix is invalid.
        """
        wavevector = real_array(wavevector, "wavevector")
        if wavevector.shape != (self.dimension,):
            raise ValueError(
                f"wavevector must have shape ({self.dimension},); "
                "use [qx] in one dimension"
            )
        fourier_force_constants = np.zeros(
            (self.num_dof, self.num_dof), dtype=np.complex128
        )
        # Keep block order and normalize after summation to preserve rounding.
        for translation_vector, force_constant_block in self.force_blocks:
            phase = np.exp(1j * np.dot(wavevector, translation_vector))
            fourier_force_constants += force_constant_block * phase
        dynamical_matrix = (
            fourier_force_constants
            / self.square_root_masses[:, None]
            / self.square_root_masses[None, :]
        )
        if not np.all(np.isfinite(dynamical_matrix)):
            raise ValueError(
                f"Nonfinite dynamical matrix at wavevector={wavevector}"
            )
        hermitian_residual = np.max(
            np.abs(dynamical_matrix - dynamical_matrix.conj().T)
        )
        if hermitian_residual > self.hermitian_tolerance:
            raise ValueError(
                f"Non-Hermitian dynamical matrix at wavevector={wavevector}"
            )
        return dynamical_matrix


def validate_model(lattice_model):
    """Validate force constants and mass-weighted translation zero modes.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.

    Returns:
        dict[str, float]: Force-constant and Gamma-point translation
            diagnostics.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    diagnostics = validate_force_constants(lattice_model)
    dynamical_matrix = build_dynamical_matrix(
        lattice_model, np.zeros(lattice_model.dimension)
    )
    # Test each Cartesian translation separately, including shear directions.
    translation_vectors = np.tile(
        np.eye(lattice_model.dimension), (lattice_model.n_atoms, 1)
    )
    translation_vectors *= np.sqrt(lattice_model.dof_masses)[:, None]
    translation_vectors /= np.linalg.norm(translation_vectors, axis=0)
    translation_residual = float(
        np.max(np.linalg.norm(dynamical_matrix @ translation_vectors, axis=0))
    )
    eigenvalue_scale = lattice_model.eigenvalue_scale
    tolerance = scaled_tolerance(eigenvalue_scale, 1e-12, 1e-10)
    if translation_residual > tolerance:
        raise ValueError(
            f"Gamma mass-weighted translation is not a zero mode: residual={translation_residual}"
        )
    diagnostics.update(
        eigenvalue_scale=eigenvalue_scale,
        translation_residual=translation_residual,
        translation_tolerance=tolerance,
    )
    return diagnostics


@dataclass
class ModeResult:
    """Store the modes and numerical diagnostics of one wavevector.

    Attributes:
        eigenvalues (np.ndarray): Unmodified squared angular frequencies.
        frequencies (np.ndarray): Angular frequencies in omega0 units.
        eigenvectors (np.ndarray): Mass-weighted unit eigenvectors as columns.
        clipped_negative (np.ndarray): Flags for small negative eigenvalues
            replaced by zero only when converting to frequencies.
        diagnostics (dict): Tolerances, scales, Hermiticity, eigen-equation,
            and orthogonality residuals.
    """

    eigenvalues: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    clipped_negative: np.ndarray
    diagnostics: dict


def solve_modes(
    dynamical_matrix, *, eigenvalue_scale=None, atol=1e-12, rtol=1e-10
):
    """Solve a Hermitian normal-mode problem without masking instabilities.

    Original eigenvalues are retained. Only small negative eigenvalues
    within the scaled tolerance are set to zero for frequency conversion.
    Positive small eigenvalues are preserved. The input matrix is never
    symmetrized automatically.

    Args:
        dynamical_matrix (array_like): Finite square matrix in omega0
            squared units; Hermiticity is checked before diagonalization.
        eigenvalue_scale (float or None): Common squared-frequency scale;
            None selects the matrix spectral norm.
        atol (float): Nonnegative absolute tolerance in the units of the
            checked quantity.
        rtol (float): Nonnegative dimensionless relative tolerance.

    Returns:
        ModeResult: Original eigenvalues, frequencies, eigenvectors, small-
            negative flags, and residual diagnostics.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    dynamical_matrix = np.asarray(dynamical_matrix)
    if (
        dynamical_matrix.ndim != 2
        or dynamical_matrix.shape[0] == 0
        or dynamical_matrix.shape[0] != dynamical_matrix.shape[1]
        or (dynamical_matrix.dtype.kind not in "fiuc")
        or (not np.all(np.isfinite(dynamical_matrix)))
    ):
        raise ValueError(
            "The dynamical matrix must be a nonempty finite numerical square matrix"
        )
    dynamical_matrix = dynamical_matrix.astype(np.complex128)
    eigenvalue_scale = (
        float(np.linalg.norm(dynamical_matrix, ord=2))
        if eigenvalue_scale is None
        else eigenvalue_scale
    )
    eigenvalue_tolerance = scaled_tolerance(eigenvalue_scale, atol, rtol)
    hermitian_residual = float(
        np.max(np.abs(dynamical_matrix - dynamical_matrix.conj().T))
    )
    if hermitian_residual > eigenvalue_tolerance:
        raise ValueError(
            f"Non-Hermitian dynamical matrix: residual={hermitian_residual}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)
    unstable = np.flatnonzero(eigenvalues < -eigenvalue_tolerance)
    if unstable.size:
        branch_index = unstable[0]
        raise ValueError(
            f"Dynamical instability: branch={branch_index}, lambda={eigenvalues[branch_index]}, tolerance={eigenvalue_tolerance}"
        )
    # Keep raw eigenvalues for diagnostics and preserve tiny positive modes.
    clipped_negative = eigenvalues < 0
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0))
    eigen_equation_residuals = (
        dynamical_matrix @ eigenvectors - eigenvectors * eigenvalues[None, :]
    )
    equation_residual = float(
        np.max(np.linalg.norm(eigen_equation_residuals, axis=0))
    )
    orthogonality_residual = float(
        np.max(
            np.abs(
                eigenvectors.conj().T @ eigenvectors - np.eye(len(eigenvalues))
            )
        )
    )
    if equation_residual > eigenvalue_tolerance:
        raise ValueError(
            f"Eigen-equation residual exceeds tolerance: residual={equation_residual}"
        )
    if orthogonality_residual > 1e-10:
        raise ValueError(
            f"Eigenvectors are not orthogonal: residual={orthogonality_residual}"
        )
    denominator = float(eigenvalue_scale) + float(atol)
    scaled_equation_residual = (
        equation_residual / denominator if denominator > 0 else 0.0
    )
    return ModeResult(
        eigenvalues=eigenvalues,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        clipped_negative=clipped_negative,
        diagnostics={
            "eigenvalue_scale": float(eigenvalue_scale),
            "eigenvalue_tolerance": eigenvalue_tolerance,
            "hermitian_residual": hermitian_residual,
            "eigen_equation_residual": equation_residual,
            "scaled_eigen_equation_residual": scaled_equation_residual,
            "orthogonality_residual": orthogonality_residual,
        },
    )


def make_monatomic_chain(mass=1.0, spring_constant=1.0, lattice_constant=1.0):
    """Create a longitudinal nearest-neighbor monatomic chain.

    Args:
        mass (float): Positive atomic mass in fixed m0 reference units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        LatticeModel: One atom and one longitudinal displacement per cell.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    parameters = real_array(
        [mass, spring_constant, lattice_constant],
        "mass, spring_constant, lattice_constant",
    )
    if parameters.shape != (3,) or np.any(parameters <= 0):
        raise ValueError(
            "mass, spring_constant, and lattice_constant must be positive real scalars"
        )
    mass, spring_constant, lattice_constant = parameters
    return LatticeModel(
        lattice_vectors=[[lattice_constant]],
        basis_positions=[[0.0]],
        masses=[mass],
        force_constants={
            (0,): [[2 * spring_constant]],
            (1,): [[-spring_constant]],
            (-1,): [[-spring_constant]],
        },
    )


def make_diatomic_chain(
    mass_1=1.0, mass_2=2.0, spring_constant=1.0, lattice_constant=1.0
):
    """Create a longitudinal diatomic chain from force-constant blocks.

    Args:
        mass_1 (float): Positive mass of the first basis atom, in m0 units.
        mass_2 (float): Positive mass of the second basis atom, in m0 units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        LatticeModel: Two atoms separated by half the cell length.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    parameters = real_array(
        [mass_1, mass_2, spring_constant, lattice_constant],
        "mass_1, mass_2, spring_constant, lattice_constant",
    )
    if parameters.shape != (4,) or np.any(parameters <= 0):
        raise ValueError(
            "mass_1, mass_2, spring_constant, and lattice_constant must be positive real scalars"
        )
    mass_1, mass_2, spring_constant, lattice_constant = parameters
    return LatticeModel(
        lattice_vectors=[[lattice_constant]],
        basis_positions=[[0.0], [lattice_constant / 2]],
        masses=[mass_1, mass_2],
        force_constants={
            (0,): [
                [2 * spring_constant, -spring_constant],
                [-spring_constant, 2 * spring_constant],
            ],
            (1,): [[0.0, 0.0], [-spring_constant, 0.0]],
            (-1,): [[0.0, -spring_constant], [0.0, 0.0]],
        },
    )


def main():
    """Print representative modes and model-validation diagnostics.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    for model_name, lattice_model in [
        ("Monatomic chain", make_monatomic_chain()),
        ("Diatomic chain", make_diatomic_chain()),
    ]:
        print(f"\n{model_name}")
        print(
            f"Spatial dimension={lattice_model.dimension}, atoms per cell={lattice_model.n_atoms}, degrees of freedom={lattice_model.n_dof}"
        )
        print("Model diagnostics: ", validate_model(lattice_model))
        lattice_constant = lattice_model.lattice_vectors[0, 0]
        for normalized_wavevector in [0.0, 0.5, 1.0]:
            dynamical_matrix = build_dynamical_matrix(
                lattice_model,
                [normalized_wavevector * np.pi / lattice_constant],
            )
            mode_result = solve_modes(
                dynamical_matrix,
                eigenvalue_scale=lattice_model.eigenvalue_scale,
            )
            print(
                f"  qa/pi={normalized_wavevector:.2f}, lambda={mode_result.eigenvalues}, omega={mode_result.frequencies}"
            )


if __name__ == "__main__":
    main()
