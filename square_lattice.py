"""Compare square lattices with axial and diagonal central springs.

The zero-diagonal-coupling model has harmonic shear zero modes. Positive
diagonal coupling restores shear stiffness. Only in-plane motion is included.
"""

from dataclasses import dataclass

import numpy as np

from visualization import save_figures
from dynamical_matrix import (
    LatticeModel,
    build_dynamical_matrix,
    real_array,
    solve_modes,
)
from dispersion import calculate_dispersion


def square_parameters(
    mass,
    nearest_neighbor_spring_constant,
    lattice_constant,
    diagonal_spring_constant,
):
    """Validate axial and diagonal square-lattice model parameters.

    Args:
        mass (float): Positive atomic mass in fixed m0 reference units.
        nearest_neighbor_spring_constant (float): Positive axial nearest-
            neighbor stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.
        diagonal_spring_constant (float): Nonnegative diagonal stiffness in
            K0 units; zero recovers the soft model.

    Returns:
        np.ndarray: Validated mass, axial stiffness, lattice constant, and
            diagonal stiffness.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    parameters = real_array(
        [
            mass,
            nearest_neighbor_spring_constant,
            lattice_constant,
            diagonal_spring_constant,
        ],
        "mass, nearest_neighbor_spring_constant, lattice_constant, diagonal_spring_constant",
    )
    if (
        parameters.shape != (4,)
        or np.any(parameters[:3] <= 0)
        or parameters[3] < 0
    ):
        raise ValueError(
            "mass, nearest-neighbor stiffness, and lattice constant must be positive; diagonal stiffness must be nonnegative"
        )
    return parameters


def make_square_lattice(
    mass=1.0,
    nearest_neighbor_spring_constant=1.0,
    lattice_constant=1.0,
    diagonal_spring_constant=0.0,
):
    """Create a square lattice with axial and optional diagonal springs.

    Args:
        mass (float): Positive atomic mass in fixed m0 reference units.
        nearest_neighbor_spring_constant (float): Positive axial nearest-
            neighbor stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.
        diagonal_spring_constant (float): Nonnegative diagonal stiffness in
            K0 units; zero recovers the soft model.

    Returns:
        LatticeModel: One atom with two in-plane displacement components.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    (
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    ) = square_parameters(
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    )
    horizontal_direction = np.array([1.0, 0.0])
    vertical_direction = np.array([0.0, 1.0])
    # A central spring contributes K * outer(direction, direction).
    horizontal_stiffness = nearest_neighbor_spring_constant * np.outer(
        horizontal_direction, horizontal_direction
    )
    vertical_stiffness = nearest_neighbor_spring_constant * np.outer(
        vertical_direction, vertical_direction
    )
    force_constants = {
        (0, 0): 2 * horizontal_stiffness + 2 * vertical_stiffness,
        (1, 0): -horizontal_stiffness,
        (-1, 0): -horizontal_stiffness,
        (0, 1): -vertical_stiffness,
        (0, -1): -vertical_stiffness,
    }
    if diagonal_spring_constant > 0:
        # Unit diagonal directions introduce the factor of one half.
        positive_diagonal_stiffness = (
            diagonal_spring_constant / 2 * np.array([[1.0, 1.0], [1.0, 1.0]])
        )
        negative_diagonal_stiffness = (
            diagonal_spring_constant / 2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        )
        force_constants[0, 0] += 2 * (
            positive_diagonal_stiffness + negative_diagonal_stiffness
        )
        force_constants[1, 1] = -positive_diagonal_stiffness
        force_constants[-1, -1] = -positive_diagonal_stiffness
        force_constants[1, -1] = -negative_diagonal_stiffness
        force_constants[-1, 1] = -negative_diagonal_stiffness
    return LatticeModel(
        lattice_vectors=[[lattice_constant, 0.0], [0.0, lattice_constant]],
        basis_positions=[[0.0, 0.0]],
        masses=[mass],
        force_constants=force_constants,
    )


def explicit_square_matrix(
    wavevector,
    mass,
    nearest_neighbor_spring_constant,
    lattice_constant,
    diagonal_spring_constant=0.0,
):
    """Evaluate an independent closed-form square-lattice matrix.

    Args:
        wavevector (array_like): Cartesian wavevector of shape (dimension,),
            in inverse a0 units.
        mass (float): Positive atomic mass in fixed m0 reference units.
        nearest_neighbor_spring_constant (float): Positive axial nearest-
            neighbor stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.
        diagonal_spring_constant (float): Nonnegative diagonal stiffness in
            K0 units; zero recovers the soft model.

    Returns:
        np.ndarray: A real symmetric (2, 2) reference matrix.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    (
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    ) = square_parameters(
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    )
    wavevector = real_array(wavevector, "wavevector")
    if wavevector.shape != (2,):
        raise ValueError("The Cartesian wavevector must have shape (2,)")
    phase_x, phase_y = wavevector * lattice_constant
    diagonal_phase_factor = (
        np.sin((phase_x + phase_y) / 2) ** 2
        + np.sin((phase_x - phase_y) / 2) ** 2
    )
    xx_component = (
        2
        / mass
        * (
            2 * nearest_neighbor_spring_constant * np.sin(phase_x / 2) ** 2
            + diagonal_spring_constant * diagonal_phase_factor
        )
    )
    yy_component = (
        2
        / mass
        * (
            2 * nearest_neighbor_spring_constant * np.sin(phase_y / 2) ** 2
            + diagonal_spring_constant * diagonal_phase_factor
        )
    )
    xy_component = (
        2 * diagonal_spring_constant / mass * np.sin(phase_x) * np.sin(phase_y)
    )
    return np.array(
        [[xx_component, xy_component], [xy_component, yy_component]]
    )


@dataclass
class SquarePath:
    """Store a Gamma-X-M-Gamma path with physical cumulative distance.

    Attributes:
        q_points (np.ndarray): Cartesian wavevectors of shape (Nq, 2).
        distances (np.ndarray): Cumulative wavevector distance in inverse a0 units.
        tick_indices (np.ndarray): Indices of Gamma, X, M, and final Gamma.
        labels (tuple[str, ...]): Math-text endpoint labels used in plots.
    """

    q_points: np.ndarray
    distances: np.ndarray
    tick_indices: np.ndarray
    labels: tuple[str, ...]


def build_square_path(lattice_constant=1.0, points_per_axis=101):
    """Construct the Gamma-X-M-Gamma path with cumulative distance.

    Common segment endpoints appear only once. The closing Gamma point is
    retained. Segment interval counts scale with Cartesian length, so the
    final diagonal segment is approximately sqrt(2) times longer.

    Args:
        lattice_constant (float): Positive primitive-cell length in a0
            units.
        points_per_axis (int): Points on an axial path segment, including
            both endpoints; at least two.

    Returns:
        SquarePath: Wavevectors, physical cumulative distances, endpoint
            indices, and labels without duplicated neighboring endpoints.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    lattice_constant = real_array(lattice_constant, "a")
    if lattice_constant.shape != () or lattice_constant <= 0:
        raise ValueError(
            "lattice_constant must be a positive finite real scalar"
        )
    if (
        not isinstance(points_per_axis, (int, np.integer))
        or points_per_axis < 2
    ):
        raise ValueError("points_per_axis must be an integer >= 2")
    lattice_constant = float(lattice_constant)
    vertices = (
        np.pi
        / lattice_constant
        * np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    )
    target_spacing = np.pi / lattice_constant / (points_per_axis - 1)
    path_segments = [vertices[:1]]
    tick_indices = [0]
    for segment_start, segment_end in zip(vertices[:-1], vertices[1:]):
        num_intervals = max(
            1,
            int(
                round(
                    np.linalg.norm(segment_end - segment_start) / target_spacing
                )
            ),
        )
        segment = np.linspace(segment_start, segment_end, num_intervals + 1)
        # Keep each shared high-symmetry endpoint only once.
        path_segments.append(segment[1:])
        tick_indices.append(tick_indices[-1] + num_intervals)
    wavevectors = np.concatenate(path_segments)
    step_lengths = np.linalg.norm(np.diff(wavevectors, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(step_lengths)))
    return SquarePath(
        wavevectors,
        distances,
        np.array(tick_indices),
        ("$\\Gamma$", "$X$", "$M$", "$\\Gamma$"),
    )


def validate_square_versions(
    mass,
    nearest_neighbor_spring_constant,
    lattice_constant,
    diagonal_spring_constant,
):
    """Validate soft and stabilized square lattices and their limits.

    Args:
        mass (float): Positive atomic mass in fixed m0 reference units.
        nearest_neighbor_spring_constant (float): Positive axial nearest-
            neighbor stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.
        diagonal_spring_constant (float): Nonnegative diagonal stiffness in
            K0 units; zero recovers the soft model.

    Returns:
        dict: Matrix errors, sampled positive-eigenvalue bound, and zero-
            diagonal-coupling convergence errors.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    if diagonal_spring_constant <= 0:
        raise ValueError(
            "Stabilized-model validation requires positive diagonal coupling; the constructor also supports zero"
        )
    nearest_neighbor_model = make_square_lattice(
        mass, nearest_neighbor_spring_constant, lattice_constant
    )
    stabilized_model = make_square_lattice(
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    )
    wavevector_axis = np.linspace(
        -np.pi / lattice_constant, np.pi / lattice_constant, 8, endpoint=False
    )
    wavevector_axis[4] = 0.0
    wavevector_x_grid, wavevector_y_grid = np.meshgrid(
        wavevector_axis, wavevector_axis, indexing="ij"
    )
    sample_wavevectors = np.column_stack(
        (wavevector_x_grid.ravel(), wavevector_y_grid.ravel())
    )
    sample_wavevectors = np.vstack(
        (
            sample_wavevectors,
            np.array([[0.23, 0.47], [-0.6, 0.4]]) / lattice_constant,
        )
    )
    nearest_neighbor_dispersion = calculate_dispersion(
        nearest_neighbor_model, sample_wavevectors
    )
    stabilized_dispersion = calculate_dispersion(
        stabilized_model, sample_wavevectors
    )
    max_matrix_error = 0.0
    for validation_model, coupling in [
        (nearest_neighbor_model, 0.0),
        (stabilized_model, diagonal_spring_constant),
    ]:
        for wavevector in sample_wavevectors:
            dynamical_matrix = build_dynamical_matrix(
                validation_model, wavevector
            )
            reference_matrix = explicit_square_matrix(
                wavevector,
                mass,
                nearest_neighbor_spring_constant,
                lattice_constant,
                coupling,
            )
            max_matrix_error = max(
                max_matrix_error,
                float(np.max(np.abs(dynamical_matrix - reference_matrix))),
            )
            np.testing.assert_allclose(
                dynamical_matrix, reference_matrix, atol=1e-12, rtol=0
            )
            swapped_matrix = build_dynamical_matrix(
                validation_model, wavevector[::-1]
            )
            np.testing.assert_allclose(
                np.linalg.eigvalsh(dynamical_matrix),
                np.linalg.eigvalsh(swapped_matrix),
                atol=1e-12,
                rtol=0,
            )
    non_gamma = np.any(sample_wavevectors != 0.0, axis=1)
    np.testing.assert_allclose(
        stabilized_dispersion.eigenvalues[~non_gamma], 0.0, atol=1e-12, rtol=0
    )
    minimum_positive = float(
        np.min(stabilized_dispersion.eigenvalues[non_gamma])
    )
    assert (
        minimum_positive > 0
    ), "The stabilized model has a sampled non-Gamma zero or unstable mode"
    assert (
        np.min(
            stabilized_dispersion.eigenvalues
            - nearest_neighbor_dispersion.eigenvalues
        )
        > -1e-12
    )
    nearest_reference = np.sort(
        4
        * nearest_neighbor_spring_constant
        / mass
        * np.sin(sample_wavevectors * lattice_constant / 2) ** 2,
        axis=1,
    )
    np.testing.assert_allclose(
        nearest_neighbor_dispersion.eigenvalues,
        nearest_reference,
        atol=1e-12,
        rtol=0,
    )
    limit_errors = []
    for coupling_ratio in [0.01, 0.0001, 1e-06]:
        small_diagonal_spring_constant = (
            coupling_ratio * nearest_neighbor_spring_constant
        )
        small_coupling_model = make_square_lattice(
            mass,
            nearest_neighbor_spring_constant,
            lattice_constant,
            small_diagonal_spring_constant,
        )
        small_coupling_dispersion = calculate_dispersion(
            small_coupling_model, sample_wavevectors
        )
        error = float(
            np.max(
                np.abs(
                    small_coupling_dispersion.eigenvalues
                    - nearest_neighbor_dispersion.eigenvalues
                )
            )
        )
        assert (
            error <= 4 * small_diagonal_spring_constant / mass + 1e-12
        ), "The zero-diagonal-coupling limit exceeds the perturbation bound"
        limit_errors.append(error)
    assert np.all(
        np.diff(limit_errors) < 0
    ), "Decreasing diagonal coupling did not recover the nearest-neighbor model"
    long_wavelength_wavevector = 0.01 / lattice_constant
    modes = solve_modes(
        build_dynamical_matrix(
            stabilized_model, [long_wavelength_wavevector, 0.0]
        ),
        eigenvalue_scale=stabilized_model.eigenvalue_scale,
    )
    measured_speeds = modes.frequencies / long_wavelength_wavevector
    expected_speeds = lattice_constant * np.sqrt(
        np.array(
            [
                diagonal_spring_constant,
                nearest_neighbor_spring_constant + diagonal_spring_constant,
            ]
        )
        / mass
    )
    np.testing.assert_allclose(
        measured_speeds, expected_speeds, atol=0, rtol=0.0001
    )
    print(
        f"[PASS] Explicit matrices, axis-exchange symmetry, and stability: m={mass:g}, K1={nearest_neighbor_spring_constant:g}, K2={diagonal_spring_constant:g}, a={lattice_constant:g}; maximum matrix error={max_matrix_error:.3e}"
    )
    print(
        f"       Minimum sampled non-Gamma eigenvalue={minimum_positive:.6f}; K2/K1=1e-2/1e-4/1e-6 limit errors={limit_errors}"
    )
    print(
        f"[PASS] Long-wavelength numerical cT/cL={measured_speeds}, reference={expected_speeds}"
    )
    return {
        "matrix_error": max_matrix_error,
        "minimum_positive_eigenvalue": minimum_positive,
        "limit_errors": limit_errors,
    }


def main():
    """Run the example, retain physical checks, and save PNG figures.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    import matplotlib.pyplot as plt

    mass = 1.0
    nearest_neighbor_spring_constant = 1.0
    diagonal_spring_constant = 0.25
    lattice_constant = 1.0
    points_per_axis = 101
    nearest_neighbor_model = make_square_lattice(
        mass, nearest_neighbor_spring_constant, lattice_constant
    )
    wavevector_path = build_square_path(lattice_constant, points_per_axis)
    nearest_neighbor_dispersion = calculate_dispersion(
        nearest_neighbor_model, wavevector_path.q_points
    )
    directional_eigenvalues = (
        4
        * nearest_neighbor_spring_constant
        / mass
        * np.sin(wavevector_path.q_points * lattice_constant / 2) ** 2
    )
    reference_eigenvalues = np.sort(directional_eigenvalues, axis=1)
    reference_frequencies = np.sqrt(reference_eigenvalues)
    path_error = float(
        np.max(
            np.abs(
                nearest_neighbor_dispersion.eigenvalues - reference_eigenvalues
            )
        )
    )
    assert (
        path_error < 1e-12
    ), "Numerical path eigenvalues do not match the analytic reference"
    expected_special_eigenvalues = (
        4
        * nearest_neighbor_spring_constant
        / mass
        * np.array([[0, 0], [0, 1], [1, 1], [0, 0]])
    )
    np.testing.assert_allclose(
        nearest_neighbor_dispersion.eigenvalues[wavevector_path.tick_indices],
        expected_special_eigenvalues,
        atol=1e-12,
        rtol=0,
    )
    for point_label, index in zip(
        ["Γ", "X", "M", "Γ"], wavevector_path.tick_indices
    ):
        print(
            f"{point_label} point: q={wavevector_path.q_points[index]}, lambda={nearest_neighbor_dispersion.eigenvalues[index]}"
        )
    gamma_to_x_slice = slice(0, wavevector_path.tick_indices[1] + 1)
    zero_mode_error = float(
        np.max(
            np.abs(nearest_neighbor_dispersion.eigenvalues[gamma_to_x_slice, 0])
        )
    )
    assert (
        zero_mode_error < 1e-12
    ), "The Gamma-X zero-frequency branch was not recovered"
    midpoint_dynamical_matrix = build_dynamical_matrix(
        nearest_neighbor_model, [np.pi / (2 * lattice_constant), 0.0]
    )
    np.testing.assert_allclose(
        midpoint_dynamical_matrix @ [0.0, 1.0], 0.0, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        nearest_neighbor_dispersion.eigenvalues[
            wavevector_path.tick_indices[2] :, 0
        ],
        nearest_neighbor_dispersion.eigenvalues[
            wavevector_path.tick_indices[2] :, 1
        ],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        wavevector_path.distances[wavevector_path.tick_indices]
        * lattice_constant
        / np.pi,
        [0.0, 1.0, 2.0, 2.0 + np.sqrt(2)],
        atol=1e-12,
        rtol=0,
    )
    assert np.all(
        np.linalg.norm(np.diff(wavevector_path.q_points, axis=0), axis=1) > 0
    )
    stabilized_model = make_square_lattice(
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    )
    stabilized_dispersion = calculate_dispersion(
        stabilized_model, wavevector_path.q_points
    )
    reference_matrices = np.array(
        [
            explicit_square_matrix(
                wavevector,
                mass,
                nearest_neighbor_spring_constant,
                lattice_constant,
                diagonal_spring_constant,
            )
            for wavevector in wavevector_path.q_points
        ]
    )
    stabilized_reference_eigenvalues = np.linalg.eigvalsh(reference_matrices)
    stabilized_path_error = float(
        np.max(
            np.abs(
                stabilized_dispersion.eigenvalues
                - stabilized_reference_eigenvalues
            )
        )
    )
    assert (
        stabilized_path_error < 1e-12
    ), "Stabilized path eigenvalues do not match the explicit matrix"
    assert np.min(stabilized_reference_eigenvalues) >= -1e-12
    stabilized_reference_frequencies = np.sqrt(
        np.maximum(stabilized_reference_eigenvalues, 0.0)
    )
    expected_stabilized_eigenvalues = np.array(
        [
            [0.0, 0.0],
            [
                4 * diagonal_spring_constant / mass,
                4
                * (nearest_neighbor_spring_constant + diagonal_spring_constant)
                / mass,
            ],
            [
                4 * nearest_neighbor_spring_constant / mass,
                4 * nearest_neighbor_spring_constant / mass,
            ],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        stabilized_dispersion.eigenvalues[wavevector_path.tick_indices],
        expected_stabilized_eigenvalues,
        atol=1e-12,
        rtol=0,
    )
    assert np.all(
        stabilized_dispersion.eigenvalues[
            1 : wavevector_path.tick_indices[1] + 1, 0
        ]
        > 0
    )
    print(
        f"[PASS] Stabilized Gamma/X/M and path: maximum eigenvalue error={stabilized_path_error:.3e}"
    )
    print(
        "Stabilized Gamma/X/M eigenvalues: ",
        stabilized_dispersion.eigenvalues[wavevector_path.tick_indices[:3]],
    )
    print(
        "Diagonal springs give the Gamma-X transverse branch a restoring force; Gamma retains two translation zero modes."
    )
    validate_square_versions(
        mass,
        nearest_neighbor_spring_constant,
        lattice_constant,
        diagonal_spring_constant,
    )
    validate_square_versions(2.0, 3.0, 1.5, 0.4)
    print(
        f"[PASS] Gamma/X/M and full-path reference: maximum eigenvalue error={path_error:.3e}"
    )
    print(
        f"[PASS] Gamma-X zero-frequency branch: maximum |lambda|={zero_mode_error:.3e}"
    )
    print(
        "Nearest-neighbor central springs lack shear stiffness; the zero-frequency branch is a harmonic soft mode."
    )
    print(
        f"Path point count={len(wavevector_path.q_points)}, normalized endpoint distances={wavevector_path.distances[wavevector_path.tick_indices] * lattice_constant / np.pi}"
    )
    path_coordinates = wavevector_path.distances * lattice_constant / np.pi
    tick_positions = path_coordinates[wavevector_path.tick_indices]
    figure, axis = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    colors = ["tab:blue", "tab:red"]
    markers = ["o", "s"]
    linestyles = ["-", "--"]
    sample_stride = max(1, len(path_coordinates) // 40)
    for branch_index in range(2):
        axis.plot(
            path_coordinates,
            reference_frequencies[:, branch_index],
            color=colors[branch_index],
            linestyle=linestyles[branch_index],
            label=f"Branch {branch_index + 1} analytical",
        )
        marker_slice = slice(
            branch_index * (sample_stride // 2), None, sample_stride
        )
        axis.plot(
            path_coordinates[marker_slice],
            nearest_neighbor_dispersion.frequencies[marker_slice, branch_index],
            linestyle="none",
            marker=markers[branch_index],
            markersize=4,
            markerfacecolor="white",
            color=colors[branch_index],
            label=f"Branch {branch_index + 1} numerical",
        )
    for position in tick_positions[1:-1]:
        axis.axvline(position, color="0.65", linewidth=0.8)
    axis.set_xticks(tick_positions, wavevector_path.labels)
    axis.set_xlim(tick_positions[0], tick_positions[-1])
    axis.set_ylim(
        bottom=-0.08 * np.sqrt(nearest_neighbor_spring_constant / mass)
    )
    axis.set_xlabel("Wavevector path (cumulative distance $sa/\\pi$)")
    axis.set_ylabel("$\\omega/\\omega_0$")
    axis.set_title(
        f"Square lattice: nearest-neighbor central springs\nm={mass:g}, K1={nearest_neighbor_spring_constant:g}, a={lattice_constant:g}; Nq={len(path_coordinates)}"
    )
    axis.annotate(
        "Zero-frequency\nshear mode",
        xy=(0.5, 0.0),
        xytext=(0.55, 0.65 * np.sqrt(nearest_neighbor_spring_constant / mass)),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.3"},
    )
    axis.grid(alpha=0.25)
    figure.legend(loc="outside lower center", ncol=2, fontsize=9)
    comparison_figure, comparison_axis = plt.subplots(
        figsize=(8.5, 5.7), layout="constrained"
    )
    for branch_index in range(2):
        comparison_axis.plot(
            path_coordinates,
            reference_frequencies[:, branch_index],
            color="0.55",
            linestyle=linestyles[branch_index],
            label=f"K2=0, branch {branch_index + 1}",
        )
        comparison_axis.plot(
            path_coordinates,
            stabilized_reference_frequencies[:, branch_index],
            color=colors[branch_index],
            linestyle=linestyles[branch_index],
            label=f"K2={diagonal_spring_constant:g}, branch {branch_index + 1}",
        )
        marker_slice = slice(
            branch_index * (sample_stride // 2), None, sample_stride
        )
        comparison_axis.plot(
            path_coordinates[marker_slice],
            stabilized_dispersion.frequencies[marker_slice, branch_index],
            linestyle="none",
            marker=markers[branch_index],
            markersize=4,
            markerfacecolor="white",
            color=colors[branch_index],
        )
    for position in tick_positions[1:-1]:
        comparison_axis.axvline(position, color="0.65", linewidth=0.8)
    comparison_axis.set_xticks(tick_positions, wavevector_path.labels)
    comparison_axis.set_xlim(tick_positions[0], tick_positions[-1])
    comparison_axis.set_ylim(
        bottom=-0.08 * np.sqrt(nearest_neighbor_spring_constant / mass)
    )
    comparison_axis.set_xlabel(
        "Wavevector path (cumulative distance $sa/\\pi$)"
    )
    comparison_axis.set_ylabel("$\\omega/\\omega_0$")
    comparison_axis.set_title(
        f"Diagonal-spring effect: K2=0 vs K2={diagonal_spring_constant:g}; m={mass:g}, K1={nearest_neighbor_spring_constant:g}, a={lattice_constant:g}\nLines: explicit-matrix reference; markers: numerical for K2>0"
    )
    comparison_axis.grid(alpha=0.25)
    comparison_figure.legend(loc="outside lower center", ncol=2, fontsize=9)
    save_figures(
        {
            "square_lattice_nearest_neighbor_dispersion": figure,
            "square_lattice_dispersion": comparison_figure,
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
