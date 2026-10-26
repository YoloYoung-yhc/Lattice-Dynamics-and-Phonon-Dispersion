"""Compute per-cell phonon densities of states using bounded-memory kernels.

Frequencies and Gaussian widths use fixed omega0 reference units. The DOS
integral equals the number of branches. Ordinary Gaussian broadening is
normalized on the full real axis, including its negative-frequency tail.
"""

from dataclasses import dataclass

import numpy as np

from dispersion import calculate_dispersion
from dynamical_matrix import nonnegative_scalar, real_array


def positive_integer(value, parameter_name):
    """Validate and return a positive integer.

    Args:
        value (int or float): Scalar to validate.
        parameter_name (str): Parameter label used in validation errors.

    Returns:
        int: The validated integer.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{parameter_name} must be a positive integer")
    return int(value)


def positive_scalar(value, parameter_name):
    """Validate and return a positive finite real scalar.

    Args:
        value (int or float): Scalar to validate.
        parameter_name (str): Parameter label used in validation errors.

    Returns:
        float: The validated scalar.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    value = nonnegative_scalar(value, parameter_name)
    if value == 0:
        raise ValueError(f"{parameter_name} must be greater than zero")
    return value


def mesh_shape_for(lattice_model, mesh_shape):
    """Normalize reciprocal-mesh axis counts for a lattice model.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        mesh_shape (int or sequence[int]): Points per reciprocal axis; a
            scalar is repeated along all dimensions.

    Returns:
        tuple[int, ...]: One positive axis count per spatial dimension.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    if isinstance(mesh_shape, (int, np.integer)):
        mesh_shape = (mesh_shape,) * lattice_model.dimension
    try:
        mesh_dimensions = tuple(
            (
                positive_integer(axis_size, "mesh axis count")
                for axis_size in mesh_shape
            )
        )
    except TypeError as error:
        raise ValueError(
            "mesh_shape must be an integer or a sequence of axis counts"
        ) from error
    if len(mesh_dimensions) != lattice_model.dimension:
        raise ValueError("mesh_shape length must match the spatial dimension")
    return mesh_dimensions


def generate_q_mesh(lattice_model, mesh_shape):
    """Generate a uniform reciprocal-cell mesh and normalized weights.

    Fractional reciprocal coordinates cover [-1/2, 1/2) without repeated
    boundaries. For chains and square lattices this is the first Brillouin
    zone. For oblique lattices it is an equivalent reciprocal primitive cell
    for periodic integrals.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        mesh_shape (int or sequence[int]): Points per reciprocal axis; a
            scalar is repeated along all dimensions.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cartesian wavevectors of shape (Nq,
            dimension) and weights of shape (Nq,) that sum to one.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    mesh_dimensions = mesh_shape_for(lattice_model, mesh_shape)
    fractional_axes = [
        np.arange(axis_size, dtype=np.float64) / axis_size - 0.5
        for axis_size in mesh_dimensions
    ]
    fractional_wavevectors = np.stack(
        np.meshgrid(*fractional_axes, indexing="ij"), axis=-1
    ).reshape(-1, lattice_model.dimension)
    # Row-vector lattice conventions require the inverse transpose.
    reciprocal_vectors = (
        2 * np.pi * np.linalg.inv(lattice_model.lattice_vectors).T
    )
    wavevectors = fractional_wavevectors @ reciprocal_vectors
    sample_weights = np.full(len(wavevectors), 1.0 / len(wavevectors))
    return (wavevectors, sample_weights)


@dataclass
class SampledModes:
    """Store a reciprocal-cell sampling without retaining eigenvectors.

    Attributes:
        mesh_shape (tuple[int, ...]): Wavevector counts along reciprocal axes.
        q_points (np.ndarray): Cartesian wavevectors, shape (Nq, dimension).
        weights (np.ndarray): Normalized wavevector integration weights.
        eigenvalues (np.ndarray): Original squared frequencies, shape (Nq, num_dof).
        frequencies (np.ndarray): Angular frequencies in fixed omega0 units.
    """

    mesh_shape: tuple[int, ...]
    q_points: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    frequencies: np.ndarray


def sample_modes(lattice_model, mesh_shape, *, wavevector_chunk_size=256):
    """Sample eigenvalues and frequencies in bounded wavevector batches.

    Args:
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        mesh_shape (int or sequence[int]): Points per reciprocal axis; a
            scalar is repeated along all dimensions.
        wavevector_chunk_size (int): Maximum wavevectors passed to the
            general solver in one batch.

    Returns:
        SampledModes: Mesh metadata, weights, original eigenvalues, and
            angular frequencies. Eigenvectors are not retained.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    mesh_dimensions = mesh_shape_for(lattice_model, mesh_shape)
    validated_chunk_size = positive_integer(
        wavevector_chunk_size, "wavevector_chunk_size"
    )
    wavevectors, sample_weights = generate_q_mesh(
        lattice_model, mesh_dimensions
    )
    eigenvalues = np.empty((len(wavevectors), lattice_model.n_dof))
    frequencies = np.empty_like(eigenvalues)
    for wavevector_start in range(0, len(wavevectors), validated_chunk_size):
        wavevector_stop = min(
            wavevector_start + validated_chunk_size, len(wavevectors)
        )
        try:
            dispersion_result = calculate_dispersion(
                lattice_model, wavevectors[wavevector_start:wavevector_stop]
            )
        except ValueError as error:
            raise ValueError(
                f"DOS wavevector block [{wavevector_start}:{wavevector_stop}]: {error}"
            ) from error
        eigenvalues[wavevector_start:wavevector_stop] = (
            dispersion_result.eigenvalues
        )
        frequencies[wavevector_start:wavevector_stop] = (
            dispersion_result.frequencies
        )
    return SampledModes(
        mesh_dimensions, wavevectors, sample_weights, eigenvalues, frequencies
    )


def frequency_grid(angular_frequencies):
    """Validate a strictly increasing angular-frequency grid.

    Args:
        angular_frequencies (array_like): Angular-frequency samples in fixed
            omega0 units.

    Returns:
        np.ndarray: A finite, strictly increasing one-dimensional float64
            grid.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    angular_frequencies = real_array(angular_frequencies, "angular_frequencies")
    if (
        angular_frequencies.ndim != 1
        or angular_frequencies.size < 2
        or np.any(np.diff(angular_frequencies) <= 0)
    ):
        raise ValueError(
            "angular_frequencies must be a strictly increasing one-dimensional grid with at least two points"
        )
    return angular_frequencies


def make_frequency_grid(
    maximum_frequency,
    broadening_width,
    *,
    points_per_sigma=5,
    tail_standard_deviations=5.0,
):
    """Build a frequency grid including Gaussian tails on both sides.

    Args:
        maximum_frequency (float): Nonnegative upper frequency of the
            sampled modes, in omega0 units.
        broadening_width (float): Positive Gaussian standard deviation in
            fixed omega0 units.
        points_per_sigma (int): Minimum number of frequency intervals per
            Gaussian standard deviation.
        tail_standard_deviations (float): Number of standard deviations
            included beyond each spectrum boundary.

    Returns:
        np.ndarray: A grid spanning both Gaussian tails, with spacing no
            larger than broadening_width / points_per_sigma.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    maximum_frequency = nonnegative_scalar(
        maximum_frequency, "maximum_frequency"
    )
    broadening_width = positive_scalar(broadening_width, "broadening_width")
    resolution = positive_integer(points_per_sigma, "points_per_sigma")
    tail_standard_deviations = positive_scalar(
        tail_standard_deviations, "tail_standard_deviations"
    )
    lower = -tail_standard_deviations * broadening_width
    upper = maximum_frequency + tail_standard_deviations * broadening_width
    intervals = max(
        1, int(np.ceil((upper - lower) / (broadening_width / resolution)))
    )
    return np.linspace(lower, upper, intervals + 1)


def gaussian_dos(
    mode_frequencies,
    angular_frequencies,
    broadening_width,
    sample_weights=None,
    *,
    mode_chunk_size=1024,
    frequency_chunk_size=128,
):
    """Accumulate a per-cell Gaussian DOS using bounded temporary arrays.

    Every branch uses the same normalized wavevector weights. Do not divide
    by the total mode count: the full-axis integral must equal the branch
    count. Gaussian kernels are processed in frequency and mode blocks.

    Args:
        mode_frequencies (array_like): Nonnegative frequencies with shape
            (Nq, num_branches), in omega0 units.
        angular_frequencies (array_like): Angular-frequency samples in fixed
            omega0 units.
        broadening_width (float): Positive Gaussian standard deviation in
            fixed omega0 units.
        sample_weights (array_like or None): Nonnegative weight per
            wavevector, summing to one; None uses uniform weights.
        mode_chunk_size (int): Maximum mode centers processed in one
            Gaussian block.
        frequency_chunk_size (int): Maximum frequency-grid points processed
            in one Gaussian block.

    Returns:
        np.ndarray: DOS on the supplied grid. Its full-axis integral
            approaches the number of branches, not one unless there is one
            branch.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    mode_frequencies = real_array(mode_frequencies, "mode_frequencies")
    if (
        mode_frequencies.ndim != 2
        or 0 in mode_frequencies.shape
        or np.any(mode_frequencies < 0)
    ):
        raise ValueError(
            "mode_frequencies must be a nonempty, nonnegative (Nq, nu) array"
        )
    angular_frequencies = frequency_grid(angular_frequencies)
    broadening_width = positive_scalar(broadening_width, "broadening_width")
    mode_chunk_size = positive_integer(mode_chunk_size, "mode_chunk_size")
    frequency_chunk_size = positive_integer(
        frequency_chunk_size, "frequency_chunk_size"
    )
    num_wavevectors, num_branches = mode_frequencies.shape
    if sample_weights is None:
        sample_weights = np.full(num_wavevectors, 1.0 / num_wavevectors)
    else:
        sample_weights = real_array(sample_weights, "sample_weights")
        if sample_weights.shape != (num_wavevectors,) or np.any(
            sample_weights < 0
        ):
            raise ValueError(
                "sample_weights must be a nonnegative vector matching the wavevector count"
            )
        if not np.isclose(np.sum(sample_weights), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError(
                "Sample weights must sum to one; weights are not rescaled automatically"
            )
    flattened_frequencies = mode_frequencies.ravel()
    # Each branch carries the wavevector weight, so the integral is num_dof.
    mode_weights = np.repeat(sample_weights, num_branches)
    density = np.zeros(len(angular_frequencies))
    # Bound temporary kernel memory in both frequency and mode dimensions.
    for frequency_start in range(
        0, len(angular_frequencies), frequency_chunk_size
    ):
        frequency_stop = min(
            frequency_start + frequency_chunk_size, len(angular_frequencies)
        )
        density_block = np.zeros(frequency_stop - frequency_start)
        for mode_start in range(0, len(flattened_frequencies), mode_chunk_size):
            mode_stop = min(
                mode_start + mode_chunk_size, len(flattened_frequencies)
            )
            kernel = (
                angular_frequencies[frequency_start:frequency_stop, None]
                - flattened_frequencies[None, mode_start:mode_stop]
            ) / broadening_width
            np.square(kernel, out=kernel)
            kernel *= -0.5
            np.exp(kernel, out=kernel)
            density_block += kernel @ mode_weights[mode_start:mode_stop]
        density[frequency_start:frequency_stop] = density_block / (
            np.sqrt(2 * np.pi) * broadening_width
        )
    if not np.all(np.isfinite(density)):
        raise ValueError(
            "DOS overflow; check the frequency and broadening scales"
        )
    return density


def integrate_curve(angular_frequencies, curve_values):
    """Integrate sampled curve values with the trapezoidal rule.

    Args:
        angular_frequencies (array_like): Angular-frequency samples in fixed
            omega0 units.
        curve_values (array_like): Finite curve values matching the angular-
            frequency grid.

    Returns:
        float: The signed trapezoidal integral in the grid coordinate.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    angular_frequencies = frequency_grid(angular_frequencies)
    curve_values = real_array(curve_values, "curve_values")
    if curve_values.shape != angular_frequencies.shape:
        raise ValueError(
            "Curve values and the frequency grid must have matching shapes"
        )
    return float(
        np.sum(
            (curve_values[:-1] + curve_values[1:])
            * np.diff(angular_frequencies)
            / 2
        )
    )


def monatomic_analytic_dos(angular_frequencies, mass=1.0, spring_constant=1.0):
    """Evaluate the unbroadened analytic DOS of a monatomic chain.

    The upper band-edge divergence is integrable. At zero frequency the
    right-hand limit is returned. Compare broadened numerical spectra with
    the equally broadened reference rather than the unbroadened peak.

    Args:
        angular_frequencies (array_like): Angular-frequency samples in fixed
            omega0 units.
        mass (float): Positive atomic mass in fixed m0 reference units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.

    Returns:
        np.ndarray: DOS values in inverse-frequency units. Values outside
            the band are zero; the upper band edge is infinite.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    angular_frequencies = real_array(angular_frequencies, "angular_frequencies")
    mass = positive_scalar(mass, "mass")
    spring_constant = positive_scalar(spring_constant, "spring_constant")
    maximum_frequency = 2 * np.sqrt(spring_constant / mass)
    density = np.zeros_like(angular_frequencies)
    inside = (angular_frequencies >= 0) & (
        angular_frequencies < maximum_frequency
    )
    density[inside] = 2 / (
        np.pi * np.sqrt(maximum_frequency**2 - angular_frequencies[inside] ** 2)
    )
    density[angular_frequencies == maximum_frequency] = np.inf
    return density


def broadened_monatomic_reference(
    angular_frequencies,
    broadening_width,
    mass=1.0,
    spring_constant=1.0,
    *,
    quadrature_order=512,
):
    """Broaden the analytic monatomic DOS using independent quadrature.

    Substituting Omega = omega_max * sin(theta) removes the band-edge
    singularity. Gauss-Legendre integration on theta in [0, pi/2] uses half
    the original quadrature weights. No dynamical matrix or numerical
    reciprocal mesh enters this reference.

    Args:
        angular_frequencies (array_like): Angular-frequency samples in fixed
            omega0 units.
        broadening_width (float): Positive Gaussian standard deviation in
            fixed omega0 units.
        mass (float): Positive atomic mass in fixed m0 reference units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        quadrature_order (int): Positive Gauss-Legendre order after removing
            the band-edge singularity.

    Returns:
        np.ndarray: An equally broadened analytic reference on the frequency
            grid.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    mass = positive_scalar(mass, "mass")
    spring_constant = positive_scalar(spring_constant, "spring_constant")
    integration_order = positive_integer(quadrature_order, "quadrature_order")
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        integration_order
    )
    # The transformed analytic measure is (2/pi) dtheta on [0, pi/2].
    integration_angles = (quadrature_nodes + 1) * np.pi / 4
    reference_frequencies = (
        2 * np.sqrt(spring_constant / mass) * np.sin(integration_angles)
    )
    return gaussian_dos(
        reference_frequencies[:, None],
        angular_frequencies,
        broadening_width,
        quadrature_weights / 2,
    )
