"""Validate and plot longitudinal modes of a monatomic harmonic chain.

Parameters use fixed reference units m0 = K0 = a0 = 1. Frequencies are
angular frequencies in units of omega0 = sqrt(K0 / m0).
"""

import numpy as np

from visualization import save_figures


def validate_parameters(mass, spring_constant, lattice_constant):
    """Validate the monatomic-chain physical parameters.

    Args:
        mass (float): Positive atomic mass in fixed m0 reference units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    values = np.asarray([mass, spring_constant, lattice_constant])
    if (
        values.shape != (3,)
        or values.dtype.kind not in "fiu"
        or (not np.all(np.isfinite(values)))
        or np.any(values <= 0)
    ):
        raise ValueError(
            "mass, spring_constant, and lattice_constant must be positive finite real scalars"
        )


def build_dynamical_matrix(wavevector, mass, spring_constant, lattice_constant):
    """Assemble a mass-normalized dynamical matrix at one wavevector.

    The matrix follows the primitive-cell phase convention. Its eigenvalues
    have squared-angular-frequency units; actual displacements are recovered
    from mass-weighted eigenvectors by dividing each atomic component by its
    square-root mass.

    Args:
        wavevector (float): Scalar Cartesian wavevector in inverse a0 units.
        mass (float): Positive atomic mass in fixed m0 reference units.
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
    validate_parameters(mass, spring_constant, lattice_constant)
    if (
        not np.isscalar(wavevector)
        or not np.isrealobj(wavevector)
        or (not np.isfinite(wavevector))
    ):
        raise ValueError("wavevector must be a finite real scalar")
    matrix_element = (
        2 * spring_constant
        - spring_constant * np.exp(-1j * wavevector * lattice_constant)
        - spring_constant * np.exp(1j * wavevector * lattice_constant)
    ) / mass
    return np.array([[matrix_element]], dtype=np.complex128)


def numerical_spectrum(wavevectors, mass, spring_constant, lattice_constant):
    """Compute raw eigenvalues and frequencies of the monatomic chain.

    Args:
        wavevectors (array_like): One-dimensional Cartesian wavevector
            samples, shape (Nq,), in inverse a0 units.
        mass (float): Positive atomic mass in fixed m0 reference units.
        spring_constant (float): Positive nearest-neighbor longitudinal
            stiffness in K0 units.
        lattice_constant (float): Positive primitive-cell length in a0
            units.

    Returns:
        tuple[np.ndarray, np.ndarray]: Raw squared frequencies and angular
            frequencies, each of shape (Nq,).

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
    """
    validate_parameters(mass, spring_constant, lattice_constant)
    wavevectors = np.asarray(wavevectors)
    if wavevectors.ndim != 1:
        raise ValueError("wavevectors must be a one-dimensional array")
    eigenvalues = np.empty(len(wavevectors))
    tolerance = 1e-12 + 1e-10 * (4 * spring_constant / mass)
    for wavevector_index, wavevector in enumerate(wavevectors):
        dynamical_matrix = build_dynamical_matrix(
            wavevector, mass, spring_constant, lattice_constant
        )
        if (
            np.max(np.abs(dynamical_matrix - dynamical_matrix.conj().T))
            > tolerance
        ):
            raise ValueError(
                f"Non-Hermitian dynamical matrix at wavevector={wavevector}"
            )
        eigenvalues[wavevector_index] = np.linalg.eigvalsh(dynamical_matrix)[0]
    unstable = np.flatnonzero(eigenvalues < -tolerance)
    if unstable.size:
        wavevector_index = unstable[0]
        raise ValueError(
            f"Negative eigenvalue at wavevector={wavevectors[wavevector_index]}, branch=0, lambda={eigenvalues[wavevector_index]}"
        )
    # Preserve positive eigenvalues; clip only roundoff-sized negatives.
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0))
    return (eigenvalues, frequencies)


def main():
    """Run the example, retain physical checks, and save PNG figures.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    import matplotlib.pyplot as plt

    num_atoms = 100
    num_wavevectors = 401
    lattice_constant = 1.0
    mass = 1.0
    spring_constant = 1.0
    validate_parameters(mass, spring_constant, lattice_constant)
    if not isinstance(num_atoms, int) or num_atoms < 3:
        raise ValueError(
            "num_atoms must be an integer >= 3 in this periodic-chain example"
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
    analytic_frequencies = (
        2
        * np.sqrt(spring_constant / mass)
        * np.abs(np.sin(wavevectors * lattice_constant / 2))
    )
    # Periodic neighbors each contribute K to the onsite force constant.
    force_constants = np.zeros((num_atoms, num_atoms))
    for atom_index in range(num_atoms):
        force_constants[atom_index, atom_index] = 2 * spring_constant
        force_constants[atom_index, (atom_index - 1) % num_atoms] = (
            -spring_constant
        )
        force_constants[atom_index, (atom_index + 1) % num_atoms] = (
            -spring_constant
        )
    eigenvalues, numerical_frequencies = numerical_spectrum(
        wavevectors, mass, spring_constant, lattice_constant
    )
    np.testing.assert_allclose(
        force_constants, force_constants.T, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        force_constants.sum(axis=1), 0.0, atol=1e-12, rtol=0
    )
    real_space_dynamical_matrix = force_constants / mass
    finite_eigenvalues = np.linalg.eigvalsh(real_space_dynamical_matrix)
    # Finite-chain allowed wavevectors differ from the plotting grid.
    mode_indices = np.arange(num_atoms) - num_atoms // 2
    allowed_wavevectors = (
        2 * np.pi * mode_indices / (num_atoms * lattice_constant)
    )
    finite_reference_eigenvalues = (
        4
        * spring_constant
        / mass
        * np.sin(allowed_wavevectors * lattice_constant / 2) ** 2
    )
    finite_spectrum_error = np.max(
        np.abs(finite_eigenvalues - np.sort(finite_reference_eigenvalues))
    )
    assert (
        finite_spectrum_error < 1e-12
    ), "The finite periodic-chain spectrum does not match the analytic reference"
    absolute_error = np.abs(numerical_frequencies - analytic_frequencies)
    print(
        "Maximum absolute eigenvalue error: ",
        np.max(np.abs(eigenvalues - analytic_frequencies**2)),
    )
    print("Maximum absolute frequency error: ", np.max(absolute_error))
    print(
        f"{num_atoms}-atom periodic-chain maximum absolute eigenvalue error: ",
        finite_spectrum_error,
    )
    validation_cases = [
        ("Current parameters", mass, spring_constant, lattice_constant),
        ("Nondefault parameters", 2.0, 3.0, 1.5),
    ]
    for (
        case_label,
        validation_mass,
        validation_spring_constant,
        validation_lattice_constant,
    ) in validation_cases:
        validation_wavevectors = np.linspace(
            -np.pi / validation_lattice_constant,
            np.pi / validation_lattice_constant,
            num_wavevectors,
        )
        validation_wavevectors[num_wavevectors // 2] = 0.0
        computed_eigenvalues, computed_frequencies = numerical_spectrum(
            validation_wavevectors,
            validation_mass,
            validation_spring_constant,
            validation_lattice_constant,
        )
        reference_frequencies = (
            2
            * np.sqrt(validation_spring_constant / validation_mass)
            * np.abs(
                np.sin(validation_wavevectors * validation_lattice_constant / 2)
            )
        )
        special_wavevectors = np.array(
            [
                0.0,
                np.pi / (2 * validation_lattice_constant),
                np.pi / validation_lattice_constant,
            ]
        )
        _, special_frequencies = numerical_spectrum(
            special_wavevectors,
            validation_mass,
            validation_spring_constant,
            validation_lattice_constant,
        )
        expected_frequencies = np.array(
            [
                0.0,
                np.sqrt(2 * validation_spring_constant / validation_mass),
                2 * np.sqrt(validation_spring_constant / validation_mass),
            ]
        )
        np.testing.assert_allclose(
            special_frequencies, expected_frequencies, atol=1e-10, rtol=0
        )
        eigenvalue_error = np.max(
            np.abs(computed_eigenvalues - reference_frequencies**2)
        )
        frequency_error = np.max(
            np.abs(computed_frequencies - reference_frequencies)
        )
        assert (
            eigenvalue_error < 1e-12
        ), f"{case_label}: eigenvalue error exceeds tolerance"
        assert (
            frequency_error < 1e-10
        ), f"{case_label}: frequency error exceeds tolerance"
        np.testing.assert_allclose(
            computed_frequencies, computed_frequencies[::-1], atol=1e-10, rtol=0
        )
        for wavevector in validation_wavevectors[
            :: max(1, num_wavevectors // 16)
        ]:
            original_matrix = build_dynamical_matrix(
                wavevector,
                validation_mass,
                validation_spring_constant,
                validation_lattice_constant,
            )
            translated_matrix = build_dynamical_matrix(
                wavevector + 2 * np.pi / validation_lattice_constant,
                validation_mass,
                validation_spring_constant,
                validation_lattice_constant,
            )
            np.testing.assert_allclose(
                original_matrix, translated_matrix, atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(
            build_dynamical_matrix(
                0.0,
                validation_mass,
                validation_spring_constant,
                validation_lattice_constant,
            ),
            0.0,
            atol=1e-12,
            rtol=0,
        )
        _, heavier_frequencies = numerical_spectrum(
            validation_wavevectors,
            4 * validation_mass,
            validation_spring_constant,
            validation_lattice_constant,
        )
        _, stiffer_frequencies = numerical_spectrum(
            validation_wavevectors,
            validation_mass,
            4 * validation_spring_constant,
            validation_lattice_constant,
        )
        np.testing.assert_allclose(
            heavier_frequencies, computed_frequencies / 2, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            stiffer_frequencies, computed_frequencies * 2, atol=1e-10, rtol=0
        )
        long_wavelength_wavevector = np.array(
            [0.01 / validation_lattice_constant]
        )
        _, long_wavelength_frequencies = numerical_spectrum(
            long_wavelength_wavevector,
            validation_mass,
            validation_spring_constant,
            validation_lattice_constant,
        )
        numerical_speed = (
            long_wavelength_frequencies[0] / long_wavelength_wavevector[0]
        )
        expected_sound_speed = validation_lattice_constant * np.sqrt(
            validation_spring_constant / validation_mass
        )
        speed_relative_error = abs(numerical_speed / expected_sound_speed - 1)
        assert (
            speed_relative_error < 0.0001
        ), f"{case_label}: long-wavelength sound speed does not match the reference"
        print(
            f"[PASS] {case_label}: special points, full zone, symmetry, periodicity, translation, scaling, and sound speed"
        )
        print(
            f"       Maximum eigenvalue error={eigenvalue_error:.3e}, maximum frequency error={frequency_error:.3e}, relative sound-speed error={speed_relative_error:.3e}"
        )
    for parameter_index, parameter_name in enumerate(
        ["mass", "spring_constant", "lattice_constant"]
    ):
        for invalid_value in [0.0, -1.0, np.nan, np.inf]:
            invalid_parameters = [mass, spring_constant, lattice_constant]
            invalid_parameters[parameter_index] = invalid_value
            try:
                numerical_spectrum([0.0], *invalid_parameters)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"Invalid parameter was accepted: {parameter_name}={invalid_value}"
                )
    print("[PASS] Invalid-input checks and all physical/numerical checks")
    normalized_wavevectors = wavevectors * lattice_constant / np.pi
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(normalized_wavevectors, analytic_frequencies, label="Analytical")
    sample_stride = max(1, num_wavevectors // 40)
    axis.plot(
        normalized_wavevectors[::sample_stride],
        numerical_frequencies[::sample_stride],
        "ro",
        markersize=4,
        label="Numerical",
    )
    axis.set_xlabel("$qa/\\pi$")
    axis.set_ylabel("$\\omega/\\omega_0$")
    axis.set_title(
        f"Monatomic chain: m={mass:g}, K={spring_constant:g}, a={lattice_constant:g}, Nq={num_wavevectors}"
    )
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    error_figure, error_axis = plt.subplots(figsize=(8, 4))
    error_axis.plot(normalized_wavevectors, absolute_error)
    error_axis.set_title("Absolute frequency error")
    error_axis.set_xlabel("$qa/\\pi$")
    error_axis.set_ylabel("$|\\Delta\\omega|/\\omega_0$")
    error_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    error_axis.grid(alpha=0.3)
    error_figure.tight_layout()
    save_figures(
        {
            "monatomic_dispersion": figure,
            "monatomic_frequency_error": error_figure,
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
