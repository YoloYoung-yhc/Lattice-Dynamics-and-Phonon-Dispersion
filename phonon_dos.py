"""Run DOS benchmarks, convergence studies, and independent figure windows.

Reuse sampled modes across broadening widths and retain per-cell mode
counting. Figures are exported as PNG files under results/figures.
"""

import numpy as np

from visualization import save_figures
from dos import (
    broadened_monatomic_reference,
    gaussian_dos,
    integrate_curve,
    make_frequency_grid,
    sample_modes,
)
from dynamical_matrix import make_diatomic_chain, make_monatomic_chain
from square_lattice import make_square_lattice


def run_study(
    model_name, lattice_model, mesh_shapes, *, monatomic_parameters=None
):
    """Check DOS mode counts, mesh convergence, and broadening sensitivity.

    The smallest broadening width controls adaptive mesh refinement. All
    broadening comparisons then use the same final mesh. Normalization
    includes the negative-frequency Gaussian tail without post-hoc
    rescaling.

    Args:
        model_name (str): Human-readable label for progress reports and plot
            titles.
        lattice_model (LatticeModel): Masses, geometry, and real-space
            force-constant blocks.
        mesh_shapes (sequence[tuple[int, ...]]): Increasing mesh resolutions
            used for convergence checks.
        monatomic_parameters (tuple[float, float] or None): Reference mass
            and stiffness for the independent monatomic DOS benchmark.

    Returns:
        dict: Sampled modes, DOS curves, branch contributions, mesh studies,
            broadening curves, and reference diagnostics.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    reference_broadening_width = 0.05
    broadening_widths = [0.1, 0.05, 0.02]
    relative_curve_tolerance = 0.01
    relative_normalization_tolerance = 0.001
    sample_cache = {}
    mesh_shapes = [tuple(mesh_shape) for mesh_shape in mesh_shapes]

    def get_samples(mesh_shape):
        """Reuse cached sampled modes or calculate a missing mesh.

        Args:
            mesh_shape (int or sequence[int]): Points per reciprocal axis; a
                scalar is repeated along all dimensions.

        Returns:
            SampledModes: The cached or newly sampled mesh.
        """
        if mesh_shape not in sample_cache:
            print(
                f"\n[{model_name}] Solving full mesh {mesh_shape}, Nq={np.prod(mesh_shape)} ...",
                flush=True,
            )
            sample_cache[mesh_shape] = sample_modes(lattice_model, mesh_shape)
        return sample_cache[mesh_shape]

    for mesh_shape in mesh_shapes:
        get_samples(mesh_shape)
    # Resolve the narrowest Gaussian before comparing broadening widths.
    for refinement in range(3):
        maximum_frequency = max(
            (
                float(np.max(mesh_samples.frequencies))
                for mesh_samples in sample_cache.values()
            )
        )
        minimum_broadening_grid = make_frequency_grid(
            maximum_frequency, min(broadening_widths)
        )
        previous_samples = get_samples(mesh_shapes[-2])
        finest_samples = get_samples(mesh_shapes[-1])
        previous_density = gaussian_dos(
            previous_samples.frequencies,
            minimum_broadening_grid,
            min(broadening_widths),
            previous_samples.weights,
        )
        finest_density = gaussian_dos(
            finest_samples.frequencies,
            minimum_broadening_grid,
            min(broadening_widths),
            finest_samples.weights,
        )
        smallest_width_difference = (
            integrate_curve(
                minimum_broadening_grid,
                np.abs(finest_density - previous_density),
            )
            / lattice_model.n_dof
        )
        print(
            f"[{model_name}] sigma={min(broadening_widths):.2f}: {mesh_shapes[-2]} -> {mesh_shapes[-1]}, DOS L1 difference/nu={smallest_width_difference:.3e}",
            flush=True,
        )
        if smallest_width_difference < relative_curve_tolerance:
            break
        if refinement == 2:
            raise AssertionError(
                f"{model_name} is not converged at the smallest broadening width; refine the wavevector mesh"
            )
        mesh_shapes.append(
            tuple((2 * axis_size for axis_size in mesh_shapes[-1]))
        )
        get_samples(mesh_shapes[-1])
    finest_samples = get_samples(mesh_shapes[-1])
    maximum_frequency = max(
        (
            float(np.max(mesh_samples.frequencies))
            for mesh_samples in sample_cache.values()
        )
    )
    angular_frequencies = make_frequency_grid(
        maximum_frequency, reference_broadening_width
    )
    mesh_curves = []
    previous_density = None
    print(
        f"\n{model_name}: sigma={reference_broadening_width}, modes per primitive cell nu={lattice_model.n_dof}"
    )
    print(
        "  Mesh          Nq     Modes       DOS integral   Relative count error   Previous-mesh L1/nu"
    )
    for mesh_shape in mesh_shapes:
        mesh_samples = get_samples(mesh_shape)
        dos_values = gaussian_dos(
            mesh_samples.frequencies,
            angular_frequencies,
            reference_broadening_width,
            mesh_samples.weights,
        )
        integrated_mode_count = integrate_curve(angular_frequencies, dos_values)
        normalization_error = (
            abs(integrated_mode_count - lattice_model.n_dof)
            / lattice_model.n_dof
        )
        assert (
            normalization_error < relative_normalization_tolerance
        ), "The DOS mode count does not match the branch count"
        relative_curve_difference = (
            None
            if previous_density is None
            else integrate_curve(
                angular_frequencies, np.abs(dos_values - previous_density)
            )
            / lattice_model.n_dof
        )
        difference_text = (
            "--"
            if relative_curve_difference is None
            else f"{relative_curve_difference:.3e}"
        )
        print(
            f"  {str(mesh_shape):12s} {len(mesh_samples.q_points):6d} {mesh_samples.frequencies.size:9d} {integrated_mode_count:13.9f} {normalization_error:13.3e} {difference_text:>18s}"
        )
        mesh_curves.append(
            (
                mesh_shape,
                dos_values,
                integrated_mode_count,
                relative_curve_difference,
            )
        )
        previous_density = dos_values
    assert (
        mesh_curves[-1][3] < relative_curve_tolerance
    ), "The wavevector mesh is not converged at the reference broadening width"
    dos_values = mesh_curves[-1][1]
    # Refine only the integration grid while keeping sampled modes fixed.
    refined_angular_frequencies = np.linspace(
        angular_frequencies[0],
        angular_frequencies[-1],
        2 * len(angular_frequencies) - 1,
    )
    refined_density = gaussian_dos(
        finest_samples.frequencies,
        refined_angular_frequencies,
        reference_broadening_width,
        finest_samples.weights,
    )
    integration_change = (
        abs(
            integrate_curve(refined_angular_frequencies, refined_density)
            - integrate_curve(angular_frequencies, dos_values)
        )
        / lattice_model.n_dof
    )
    assert (
        integration_change < 0.0001
    ), "The frequency integration step is not converged"
    print(
        f"[PASS] Halved frequency step: relative integral change={integration_change:.3e}"
    )
    branch_densities = np.array(
        [
            gaussian_dos(
                finest_samples.frequencies[:, [branch_index]],
                angular_frequencies,
                reference_broadening_width,
                finest_samples.weights,
            )
            for branch_index in range(lattice_model.n_dof)
        ]
    )
    np.testing.assert_allclose(
        np.sum(branch_densities, axis=0), dos_values, atol=1e-12, rtol=0
    )
    for contribution in branch_densities:
        assert (
            abs(integrate_curve(angular_frequencies, contribution) - 1)
            < relative_normalization_tolerance
        )
    broadening_curves = {}
    for broadening_width in broadening_widths:
        frequency_grid_values = make_frequency_grid(
            maximum_frequency, broadening_width
        )
        if broadening_width == reference_broadening_width:
            # The same grid and width were evaluated in the mesh study.
            broadened = dos_values
        else:
            broadened = gaussian_dos(
                finest_samples.frequencies,
                frequency_grid_values,
                broadening_width,
                finest_samples.weights,
            )
        integrated_mode_count = integrate_curve(
            frequency_grid_values, broadened
        )
        assert (
            abs(integrated_mode_count / lattice_model.n_dof - 1)
            < relative_normalization_tolerance
        )
        broadening_curves[broadening_width] = (frequency_grid_values, broadened)
        print(
            f"[PASS] Broadening scan sigma={broadening_width:.2f}: DOS integral={integrated_mode_count:.9f}, fixed mesh={mesh_shapes[-1]}"
        )
    reference_density = None
    reference_error = None
    if monatomic_parameters is not None:
        mass, spring_constant = monatomic_parameters
        reference_density = broadened_monatomic_reference(
            angular_frequencies,
            reference_broadening_width,
            mass,
            spring_constant,
        )
        finer_reference = broadened_monatomic_reference(
            angular_frequencies,
            reference_broadening_width,
            mass,
            spring_constant,
            quadrature_order=1024,
        )
        quadrature_error = integrate_curve(
            angular_frequencies, np.abs(reference_density - finer_reference)
        )
        assert (
            quadrature_error < 1e-06
        ), "Increase the analytic-reference quadrature order"
        reference_error = integrate_curve(
            angular_frequencies, np.abs(dos_values - reference_density)
        )
        assert (
            reference_error < 0.0001
        ), "Monatomic DOS does not match the equally broadened analytic reference"
        print(
            f"[PASS] Equally broadened analytic reference: L1 error={reference_error:.3e}, reference quadrature refinement difference={quadrature_error:.3e}"
        )
    return {
        "name": model_name,
        "model": lattice_model,
        "sigma": reference_broadening_width,
        "samples": finest_samples,
        "omega": angular_frequencies,
        "density": dos_values,
        "branch_densities": branch_densities,
        "mesh_curves": mesh_curves,
        "sigma_curves": broadening_curves,
        "reference": reference_density,
        "reference_error": reference_error,
        "integration_change": integration_change,
        "smallest_sigma_difference": smallest_width_difference,
    }


def main():
    """Run the example, retain physical checks, and save PNG figures.

    Returns:
        tuple[list[dict], list[Figure]]: Completed numerical studies and
            their five independent figures.

    Raises:
        ValueError: If inputs violate the documented domain or physical
            constraints.
        AssertionError: If a physical or numerical validation check fails.
    """
    import matplotlib.pyplot as plt

    studies = [
        run_study(
            "Monatomic chain",
            make_monatomic_chain(),
            [(256,), (1024,), (4096,)],
            monatomic_parameters=(1.0, 1.0),
        ),
        run_study(
            "Diatomic chain", make_diatomic_chain(), [(256,), (1024,), (4096,)]
        ),
        run_study(
            "Stable square lattice",
            make_square_lattice(diagonal_spring_constant=0.25),
            [(32, 32), (64, 64), (128, 128)],
        ),
    ]
    square_study = studies[2]
    # In two dimensions, acoustic linear dispersion gives g(omega) ~ omega.
    long_wavelength_window = (square_study["omega"] >= 0.2) & (
        square_study["omega"] <= 0.35
    )
    low_angular_frequencies = square_study["omega"][long_wavelength_window]
    low_frequency_dos = square_study["density"][long_wavelength_window]
    linear_dos_coefficient = float(
        low_angular_frequencies
        @ low_frequency_dos
        / (low_angular_frequencies @ low_angular_frequencies)
    )
    relative_linear_fit_error = float(
        np.linalg.norm(
            low_frequency_dos - linear_dos_coefficient * low_angular_frequencies
        )
        / np.linalg.norm(low_frequency_dos)
    )
    assert (
        linear_dos_coefficient > 0 and relative_linear_fit_error < 0.1
    ), "The selected 2D low-frequency window does not support a linear approximation"
    print(
        f"\n[PASS] 2D low-frequency window 0.20<=omega/omega0<=0.35: g ~ {linear_dos_coefficient:.6f}*omega; relative fit residual={relative_linear_fit_error:.3e}"
    )
    print(
        "Negative-frequency curves are Gaussian tails. Integrals use the full grid without post-hoc rescaling."
    )
    figures = []
    colors = ["tab:blue", "tab:red"]
    for model_index, study_result in enumerate(studies):
        figure, axis = plt.subplots(figsize=(8, 5.4), layout="constrained")
        figures.append(figure)
        frequency_grid_values, dos_values = (
            study_result["omega"],
            study_result["density"],
        )
        axis.axvspan(
            frequency_grid_values[0], 0.0, color="0.9", label="Gaussian tail"
        )
        axis.plot(
            frequency_grid_values,
            dos_values,
            color="black",
            linewidth=1.8,
            label="Total DOS",
        )
        if study_result["reference"] is not None:
            axis.plot(
                frequency_grid_values,
                study_result["reference"],
                "--",
                color="tab:orange",
                label="Broadened analytic reference",
            )
        else:
            labels = (
                ["Acoustic", "Optical"]
                if model_index == 1
                else ["Branch 1", "Branch 2"]
            )
            for branch_index, label in enumerate(labels):
                axis.plot(
                    frequency_grid_values,
                    study_result["branch_densities"][branch_index],
                    "--",
                    color=colors[branch_index],
                    label=label,
                )
        if model_index == 1:
            axis.axvspan(
                1.0,
                np.sqrt(2),
                color="gold",
                alpha=0.12,
                label="Unbroadened band gap",
            )
        if model_index == 2:
            axis.plot(
                low_angular_frequencies,
                linear_dos_coefficient * low_angular_frequencies,
                color="tab:green",
                linewidth=2,
                label="Low-frequency linear fit",
            )
        integrated_mode_count = integrate_curve(
            frequency_grid_values, dos_values
        )
        axis.set_title(
            f"{study_result['name']}: DOS per primitive cell\nsigma={study_result['sigma']}, mesh={study_result['samples'].mesh_shape}, integral={integrated_mode_count:.6f}"
        )
        axis.set_xlabel("$\\omega/\\omega_0$")
        axis.set_ylabel("$\\omega_0 g(\\omega)$")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        figure.legend(loc="outside lower center", ncol=2, fontsize=9)
    broadening_figure, broadening_axis = plt.subplots(
        figsize=(8, 5), layout="constrained"
    )
    figures.append(broadening_figure)
    for broadening_width, (frequency_grid_values, dos_values) in studies[0][
        "sigma_curves"
    ].items():
        broadening_axis.plot(
            frequency_grid_values,
            dos_values,
            label=f"sigma={broadening_width:.2f}",
        )
    broadening_axis.set_title(
        f"Gaussian broadening: monatomic chain\nFixed q mesh={studies[0]['samples'].mesh_shape}"
    )
    broadening_axis.set_xlabel("$\\omega/\\omega_0$")
    broadening_axis.set_ylabel("$\\omega_0 g(\\omega)$")
    broadening_axis.set_ylim(bottom=0)
    broadening_axis.grid(alpha=0.25)
    broadening_axis.legend()
    mesh_figure, mesh_axis = plt.subplots(figsize=(8, 5), layout="constrained")
    figures.append(mesh_figure)
    for mesh_shape, dos_values, _, _ in square_study["mesh_curves"]:
        mesh_axis.plot(
            square_study["omega"],
            dos_values,
            label=" x ".join(map(str, mesh_shape)),
        )
    mesh_axis.set_title(
        "q-mesh convergence: stable square lattice\nK1=1, K2=0.25, sigma=0.05"
    )
    mesh_axis.set_xlabel("$\\omega/\\omega_0$")
    mesh_axis.set_ylabel("$\\omega_0 g(\\omega)$")
    mesh_axis.set_ylim(bottom=0)
    mesh_axis.grid(alpha=0.25)
    mesh_axis.legend(title="q mesh")
    figure_names = [
        "monatomic_dos",
        "diatomic_dos",
        "square_lattice_dos",
        "monatomic_dos_broadening_comparison",
        "square_lattice_dos_mesh_convergence",
    ]
    save_figures(dict(zip(figure_names, figures, strict=True)))
    plt.show()
    return (studies, figures)


if __name__ == "__main__":
    main()
