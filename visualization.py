"""Save named Matplotlib figures with consistent PNG export settings."""

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def save_figures(
    figures: Mapping[str, "Figure"], output_directory=None, *, dpi=300
):
    """Save each figure as a PNG file without closing its window.

    Args:
        figures: Mapping from filename stems to Matplotlib figures.
        output_directory: Destination directory. Defaults to results/figures
            next to this module.
        dpi: Resolution of the saved PNG files, in dots per inch.

    Returns:
        list[Path]: Paths of the exported PNG files, in mapping order.
    """
    if output_directory is None:
        output_directory = (
            Path(__file__).resolve().parent / "results" / "figures"
        )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for filename_stem, figure in figures.items():
        output_path = output_directory / f"{filename_stem}.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)
        print(f"Saved: {output_path.name}")
    return saved_paths
