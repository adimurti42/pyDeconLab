"""VS Code-friendly entrypoint for pyDeconLab deconvolution.

Edit the parameters in the configuration block below and run this file.
If command-line arguments are provided, they are passed through to the
package CLI unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from pydeconlab.main import main


# ---------------------------------------------------------------------------
# Editable parameters for "Run Python File" in VS Code
# ---------------------------------------------------------------------------
input_path = "C:\\DyNaMo\\pyDeconLab\\data\\Stack 11.56.13 Green.tif"
output_path = "C:\\DyNaMo\\pyDeconLab\\output\\Stack 11.56.13 Green_deconv3.tif"
algorithm = "RL"
psf_mode = "gaussian"
psf_shape = (21, 21, 21)
psf_sigma = (2.0, 2.0, 4.0)
iterations = 20
bead_stack = None
block_size = 6
force_blockwise = True
stream_output = True

# Optional extras for measured PSFs
psf_file = None


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _build_configured_argv() -> list[str]:
    input_file = _resolve_path(input_path)
    output_file = _resolve_path(output_path)
    bead_file = _resolve_path(bead_stack)
    measured_psf_file = _resolve_path(psf_file)

    if input_file is None:
        raise ValueError("input_path must be set.")

    argv = [
        "--input",
        str(input_file),
        "--algo",
        str(algorithm),
        "--psf",
        str(psf_mode),
        "--iterations",
        str(int(iterations)),
    ]

    if output_file is not None:
        argv.extend(["--output", str(output_file)])

    if psf_shape is not None:
        argv.extend(["--psf-shape", *(str(int(v)) for v in psf_shape)])

    if psf_mode == "gaussian" and psf_sigma is not None:
        argv.extend(["--psf-sigma", *(str(float(v)) for v in psf_sigma)])

    if psf_mode == "measured":
        if measured_psf_file is not None:
            argv.extend(["--psf-file", str(measured_psf_file)])
        if bead_file is not None:
            argv.extend(["--bead-stack", str(bead_file)])
        if measured_psf_file is None and bead_file is None:
            raise ValueError(
                "For psf_mode='measured', set either psf_file or bead_stack at the top of run_deconv.py."
            )

    if block_size is not None:
        argv.extend(["--block-size", str(int(block_size))])

    if force_blockwise:
        argv.append("--force-blockwise")

    if stream_output:
        argv.append("--stream-output")

    return argv


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    argv = sys.argv[1:] if len(sys.argv) > 1 else _build_configured_argv()
    raise SystemExit(main(argv))
