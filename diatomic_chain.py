"""Validate and plot acoustic and optical modes of a diatomic chain.

The primitive-cell length is a; neighboring atoms are separated by a / 2.
Eigenvectors are mass weighted and stored as columns.
"""

import numpy as np

from visualization import save_figures


def build_dynamical_matrix(
    wavevector, mass_1, mass_2, spring_constant, lattice_constant
):
    """Assemble a mass-normalized dynamical matrix at one wavevector.

    The matrix follows the primitive-cell phase convention. Its eigenvalues
    have squared-angular-frequency units; actual displacements are recovered
    from mass-weighted eigenvectors by dividing each atomic component by its
    square-root mass.

    Args:
        wavevector (float): Scalar Cartesian wavevector in inverse a0 units.
        mass_1 (float): Positive mass of the first basis atom, in m0 units.
        mass_2 (float): Positive mass of the second basis atom, in m0 units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        np.ndarray: A complex128 matrix whose eigenvalues are squared
            angular frequencies.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    parameters = np.asarray([mass_1, mass_2, spring_constant, lattice_constant])
    if (
        parameters.shape != (4,)
        or parameters.dtype.kind not in "fiu"
        or (not np.all(np.isfinite(parameters)))
        or np.any(parameters <= 0)
    ):
        raise ValueError(
            "mass_1, mass_2, spring_constant, and lattice_constant must be positive finite real scalars"
        )
    if (
        not np.isscalar(wavevector)
        or not np.isrealobj(wavevector)
        or (not np.isfinite(wavevector))
    ):
        raise ValueError("wavevector must be a finite real scalar")
    first_diagonal = 2 * spring_constant / mass_1
    second_diagonal = 2 * spring_constant / mass_2
    upper_coupling = (
        -spring_constant
        * (1 + np.exp(-1j * wavevector * lattice_constant))
        / np.sqrt(mass_1 * mass_2)
    )
    lower_coupling = (
        -spring_constant
        * (1 + np.exp(1j * wavevector * lattice_constant))
        / np.sqrt(mass_1 * mass_2)
    )
    return np.array(
        [[first_diagonal, upper_coupling], [lower_coupling, second_diagonal]],
        dtype=np.complex128,
    )


def numerical_modes(
    wavevectors, mass_1, mass_2, spring_constant, lattice_constant
):
    """Compute diatomic-chain eigenvalues, frequencies, and mode vectors.

    Args:
        wavevectors (array_like): One-dimensional Cartesian wavevector
            samples, shape (Nq,), in inverse a0 units.
        mass_1 (float): Positive mass of the first basis atom, in m0 units.
        mass_2 (float): Positive mass of the second basis atom, in m0 units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Raw eigenvalues,
            frequencies, and column-wise mass-weighted eigenvectors.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    wavevectors = np.asarray(wavevectors)
    if wavevectors.ndim != 1 or wavevectors.size == 0:
        raise ValueError("wavevectors must be a nonempty one-dimensional array")
    eigenvalues = np.empty((len(wavevectors), 2))
    eigenvectors = np.empty((len(wavevectors), 2, 2), dtype=np.complex128)
    for wavevector_index, wavevector in enumerate(wavevectors):
        dynamical_matrix = build_dynamical_matrix(
            wavevector, mass_1, mass_2, spring_constant, lattice_constant
        )
        np.testing.assert_allclose(
            dynamical_matrix, dynamical_matrix.conj().T, atol=1e-12, rtol=1e-10
        )
        eigenvalues[wavevector_index], eigenvectors[wavevector_index] = (
            np.linalg.eigh(dynamical_matrix)
        )
    tolerance = 1e-12 + 1e-10 * (
        2 * spring_constant * (1 / mass_1 + 1 / mass_2)
    )
    unstable = np.argwhere(eigenvalues < -tolerance)
    if unstable.size:
        wavevector_index, branch_index = unstable[0]
        raise ValueError(
            f"Negative eigenvalue at wavevector={wavevectors[wavevector_index]}, branch={branch_index}, lambda={eigenvalues[wavevector_index, branch_index]}"
        )
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0))
    return (eigenvalues, frequencies, eigenvectors)


def analytic_spectrum(
    wavevectors, mass_1, mass_2, spring_constant, lattice_constant
):
    """Evaluate stable analytic squared frequencies of a diatomic chain.

    Args:
        wavevectors (array_like): One-dimensional Cartesian wavevector
            samples, shape (Nq,), in inverse a0 units.
        mass_1 (float): Positive mass of the first basis atom, in m0 units.
        mass_2 (float): Positive mass of the second basis atom, in m0 units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        np.ndarray: Squared acoustic and optical frequencies with shape (Nq,
            2).

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    wavevectors = np.asarray(wavevectors)
    half_trace = spring_constant * (1 / mass_1 + 1 / mass_2)
    characteristic_determinant = (
        4
        * spring_constant**2
        / (mass_1 * mass_2)
        * np.sin(wavevectors * lattice_constant / 2) ** 2
    )
    discriminant = (
        spring_constant / mass_1 - spring_constant / mass_2
    ) ** 2 + 4 * spring_constant**2 / (mass_1 * mass_2) * np.cos(
        wavevectors * lattice_constant / 2
    ) ** 2
    optical_eigenvalues = half_trace + np.sqrt(discriminant)
    # Use the eigenvalue product to avoid cancellation near Gamma.
    acoustic_eigenvalues = characteristic_determinant / optical_eigenvalues
    return np.column_stack((acoustic_eigenvalues, optical_eigenvalues))


def main():
    """Run the example, retain physical checks, and save PNG figures.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    import matplotlib.pyplot as plt

    mass_1 = 1.0
    mass_2 = 2.0
    spring_constant = 1.0
    lattice_constant = 1.0
    num_wavevectors = 401
    gamma_matrix = build_dynamical_matrix(
        0.0, mass_1, mass_2, spring_constant, lattice_constant
    )
    assert gamma_matrix.shape == (2, 2)
    assert gamma_matrix.dtype == np.complex128
    np.testing.assert_allclose(
        gamma_matrix, gamma_matrix.conj().T, atol=1e-12, rtol=0
    )
    gamma_eigenvalues, gamma_eigenvectors = np.linalg.eigh(gamma_matrix)
    expected_gamma_eigenvalues = np.array(
        [0.0, 2 * spring_constant * (1 / mass_1 + 1 / mass_2)]
    )
    np.testing.assert_allclose(
        gamma_eigenvalues, expected_gamma_eigenvalues, atol=1e-12, rtol=1e-10
    )
    print("Gamma dynamical matrix:")
    print(gamma_matrix)
    print("Matrix shape: ", gamma_matrix.shape)
    print("Matrix dtype: ", gamma_matrix.dtype)
    print("Numerical eigenvalues: ", gamma_eigenvalues)
    print("Reference eigenvalues: ", expected_gamma_eigenvalues)
    print("[PASS] Gamma matrix shape, dtype, Hermiticity, and eigenvalues")
    masses = np.array([mass_1, mass_2])
    # Convert mass-weighted eigenvectors to physical displacements.
    gamma_displacements = gamma_eigenvectors / np.sqrt(masses)[:, None]
    acoustic_displacements = gamma_displacements[:, 0]
    optical_displacements = gamma_displacements[:, 1]
    acoustic_mode_error = abs(
        acoustic_displacements[0] - acoustic_displacements[1]
    ) / np.linalg.norm(acoustic_displacements)
    optical_mode_error = abs(masses @ optical_displacements) / np.sum(
        masses * np.abs(optical_displacements)
    )
    assert (
        acoustic_mode_error < 1e-12
    ), "The Gamma acoustic displacements are not equal and in phase"
    assert (
        optical_mode_error < 1e-12
    ), "The Gamma optical mode does not preserve the cell center of mass"
    np.testing.assert_allclose(
        gamma_matrix @ np.sqrt(masses), 0.0, atol=1e-12, rtol=0
    )
    print(
        "Gamma acoustic U2/U1: ",
        np.real_if_close(acoustic_displacements[1] / acoustic_displacements[0]),
    )
    print(
        "Gamma optical U2/U1: ",
        np.real_if_close(optical_displacements[1] / optical_displacements[0]),
    )
    print(
        f"[PASS] Gamma modes: acoustic U1=U2; optical U2/U1=-m1/m2={-mass_1 / mass_2:g}"
    )
    if (
        not isinstance(num_wavevectors, int)
        or num_wavevectors < 3
        or num_wavevectors % 2 == 0
    ):
        raise ValueError(
            "num_wavevectors must be an odd integer >= 3 to include Gamma"
        )
    wavevectors = np.linspace(
        -np.pi / lattice_constant, np.pi / lattice_constant, num_wavevectors
    )
    wavevectors[num_wavevectors // 2] = 0.0
    eigenvalues, numerical_frequencies, eigenvectors = numerical_modes(
        wavevectors, mass_1, mass_2, spring_constant, lattice_constant
    )
    analytic_eigenvalues = analytic_spectrum(
        wavevectors, mass_1, mass_2, spring_constant, lattice_constant
    )
    analytic_frequencies = np.sqrt(analytic_eigenvalues)
    absolute_error = np.abs(numerical_frequencies - analytic_frequencies)
    branch_names = ["Acoustic", "Optical"]
    for branch_index, branch_name in enumerate(
        ["Acoustic branch", "Optical branch"]
    ):
        lambda_error = np.max(
            np.abs(
                eigenvalues[:, branch_index]
                - analytic_eigenvalues[:, branch_index]
            )
        )
        frequency_error = np.max(absolute_error[:, branch_index])
        print(
            f"{branch_name}: maximum absolute eigenvalue error={lambda_error:.3e}, maximum absolute frequency error={frequency_error:.3e}"
        )
    assert (
        np.max(np.abs(eigenvalues - analytic_eigenvalues)) < 1e-12
    ), "Full-zone eigenvalue error exceeds tolerance"
    away_from_gamma = wavevectors != 0.0
    assert (
        np.max(absolute_error[away_from_gamma]) < 1e-10
    ), "Frequency error away from Gamma exceeds tolerance"
    assert (
        np.max(absolute_error[:, 1]) < 1e-10
    ), "Optical frequency error exceeds tolerance"
    np.testing.assert_allclose(
        eigenvalues, eigenvalues[::-1], atol=1e-12, rtol=0
    )
    expected_boundary_eigenvalues = np.sort(
        [2 * spring_constant / mass_1, 2 * spring_constant / mass_2]
    )
    np.testing.assert_allclose(
        eigenvalues[[0, -1]],
        np.tile(expected_boundary_eigenvalues, (2, 1)),
        atol=1e-12,
        rtol=0,
    )
    for wavevector in [0.0, 0.37 / lattice_constant, -0.81 / lattice_constant]:
        np.testing.assert_allclose(
            build_dynamical_matrix(
                wavevector + 2 * np.pi / lattice_constant,
                mass_1,
                mass_2,
                spring_constant,
                lattice_constant,
            ),
            build_dynamical_matrix(
                wavevector, mass_1, mass_2, spring_constant, lattice_constant
            ),
            atol=1e-12,
            rtol=0,
        )
    print(
        "[PASS] Full-zone errors, even symmetry, boundary eigenvalues, and reciprocal periodicity"
    )
    equal_mass_eigenvalues, equal_mass_frequencies, _ = numerical_modes(
        wavevectors, mass_1, mass_1, spring_constant, lattice_constant
    )
    primitive_spacing = lattice_constant / 2
    unfolded_wavevectors = np.column_stack(
        (wavevectors, wavevectors + 2 * np.pi / lattice_constant)
    )
    folded_reference_frequencies = (
        2
        * np.sqrt(spring_constant / mass_1)
        * np.abs(np.sin(unfolded_wavevectors * primitive_spacing / 2))
    )
    folding_error = np.max(
        np.abs(equal_mass_eigenvalues - folded_reference_frequencies**2)
    )
    assert (
        folding_error < 1e-12
    ), "Equal-mass dispersion does not match the folded monatomic chain"
    equal_mass_gap = (
        equal_mass_frequencies[-1, 1] - equal_mass_frequencies[-1, 0]
    )
    assert (
        abs(equal_mass_gap) < 1e-10
    ), "The equal-mass boundary gap did not close"
    print(
        f"[PASS] Equal-mass folding: maximum eigenvalue error={folding_error:.3e}, boundary gap={equal_mass_gap:.3e}"
    )
    mass_ratios = [1, 2, 4]
    ratio_spectra = []
    band_gaps = []
    print(
        f"Mass-ratio study with fixed m1={mass_1:g}, K={spring_constant:g}, a={lattice_constant:g}, omega0=sqrt(K0/m0)"
    )
    for mass_ratio in mass_ratios:
        comparison_second_mass = mass_ratio * mass_1
        ratio_eigenvalues, ratio_frequencies, _ = numerical_modes(
            wavevectors,
            mass_1,
            comparison_second_mass,
            spring_constant,
            lattice_constant,
        )
        ratio_reference_eigenvalues = analytic_spectrum(
            wavevectors,
            mass_1,
            comparison_second_mass,
            spring_constant,
            lattice_constant,
        )
        assert (
            np.max(np.abs(ratio_eigenvalues - ratio_reference_eigenvalues))
            < 1e-12
        ), "Mass-ratio eigenvalue error exceeds tolerance"
        expected_edge = np.sort(
            [
                2 * spring_constant / mass_1,
                2 * spring_constant / comparison_second_mass,
            ]
        )
        np.testing.assert_allclose(
            ratio_eigenvalues[-1], expected_edge, atol=1e-12, rtol=0
        )
        band_gap = ratio_frequencies[-1, 1] - ratio_frequencies[-1, 0]
        expected_gap = np.sqrt(expected_edge[1]) - np.sqrt(expected_edge[0])
        np.testing.assert_allclose(band_gap, expected_gap, atol=1e-10, rtol=0)
        ratio_spectra.append(ratio_frequencies)
        band_gaps.append(band_gap)
        print(
            f"  m2/m1={mass_ratio}: acoustic boundary frequency={ratio_frequencies[-1, 0]:.6f}, optical frequency={ratio_frequencies[-1, 1]:.6f}, gap={band_gap:.6f}"
        )
    boundary_frequencies = np.array(
        [spectrum[-1] for spectrum in ratio_spectra]
    )
    assert np.all(
        np.diff(boundary_frequencies[:, 0]) < 0
    ), "The acoustic boundary frequency did not decrease with mass_2"
    np.testing.assert_allclose(
        boundary_frequencies[:, 1],
        np.sqrt(2 * spring_constant / mass_1),
        atol=1e-10,
        rtol=0,
    )
    assert np.all(
        np.diff(band_gaps) > 0
    ), "The gap did not widen with increasing mass ratio"
    print(
        "[PASS] Increasing mass_2 lowers the acoustic boundary frequency and widens the gap; the optical boundary stays fixed"
    )
    normalized_wavevectors = wavevectors * lattice_constant / np.pi
    sample_stride = max(1, num_wavevectors // 40)
    colors = ["tab:blue", "tab:red"]
    markers = ["o", "s"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for branch_index, branch_name in enumerate(branch_names):
        axis.plot(
            normalized_wavevectors,
            analytic_frequencies[:, branch_index],
            color=colors[branch_index],
            label=f"{branch_name} analytical",
        )
        axis.plot(
            normalized_wavevectors[::sample_stride],
            numerical_frequencies[::sample_stride, branch_index],
            linestyle="none",
            marker=markers[branch_index],
            markersize=4,
            color=colors[branch_index],
            markerfacecolor="white",
            label=f"{branch_name} numerical",
        )
    axis.set_xlabel("$qa/\\pi$")
    axis.set_ylabel("$\\omega/\\omega_0$")
    axis.set_title(
        f"Diatomic chain: m1={mass_1:g}, m2={mass_2:g}, K={spring_constant:g}, a={lattice_constant:g}, Nq={num_wavevectors}"
    )
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    error_figure, error_axis = plt.subplots(figsize=(8, 4))
    for branch_index, branch_name in enumerate(branch_names):
        error_axis.plot(
            normalized_wavevectors,
            absolute_error[:, branch_index],
            color=colors[branch_index],
            label=branch_name,
        )
    error_axis.set_xlabel("$qa/\\pi$")
    error_axis.set_ylabel("$|\\Delta\\omega|/\\omega_0$")
    error_axis.set_title("Absolute frequency error")
    error_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    error_axis.legend()
    error_axis.grid(alpha=0.3)
    error_figure.tight_layout()
    ratio_figure, ratio_axis = plt.subplots(figsize=(8, 5))
    ratio_colors = ["tab:orange", "tab:blue", "tab:red"]
    for mass_ratio, spectrum, color in zip(
        mass_ratios, ratio_spectra, ratio_colors
    ):
        ratio_axis.plot(
            normalized_wavevectors,
            spectrum[:, 0],
            color=color,
            label=f"m2/m1 = {mass_ratio}",
        )
        ratio_axis.plot(
            normalized_wavevectors, spectrum[:, 1], "--", color=color
        )
    ratio_axis.set_xlabel("$qa/\\pi$")
    ratio_axis.set_ylabel("$\\omega/\\omega_0$")
    ratio_axis.set_title(
        f"Mass-ratio comparison: solid = acoustic, dashed = optical\nFixed m1={mass_1:g}, K={spring_constant:g}, a={lattice_constant:g}; fixed reference units"
    )
    ratio_axis.legend()
    ratio_axis.grid(alpha=0.3)
    ratio_figure.tight_layout()
    save_figures(
        {
            "diatomic_dispersion": figure,
            "diatomic_frequency_error": error_figure,
            "diatomic_mass_ratio_comparison": ratio_figure,
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
