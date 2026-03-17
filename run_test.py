"""Quick debug runner for the high-quality FFT Richardson-Lucy pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pydeconlab.algorithms import FastRichardsonLucy
from pydeconlab.core import deconvolve_volume
from pydeconlab.io import _generate_gaussian_psf, load_tiff, save_tiff

logger = logging.getLogger(__name__)

DEBUG_ITERATIONS = 10
DEBUG_PSF_SHAPE = (21, 21, 21)
DEBUG_PSF_SIGMA = (2.0, 2.0, 4.0)
DEBUG_BLOCK_SIZE = 24


def _find_first_tiff(data_dir: Path) -> Path:
    tif_files = sorted(list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in {data_dir}")
    return tif_files[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = _find_first_tiff(data_dir)
    logger.info("Loading image stack from %s", input_path)
    stack = load_tiff(input_path, memmap=True)

    crop_depth = min(24, stack.shape[0])
    crop_y = min(256, stack.shape[1])
    crop_x = min(256, stack.shape[2])
    logger.info("Cropping debug volume to z=%d, y=%d, x=%d", crop_depth, crop_y, crop_x)
    cropped = np.asarray(stack[:crop_depth, :crop_y, :crop_x], dtype=np.float32)

    logger.info("Generating Gaussian PSF...")
    psf = _generate_gaussian_psf(shape=DEBUG_PSF_SHAPE, sigma=DEBUG_PSF_SIGMA)

    deconvolver = FastRichardsonLucy(psf=psf, iterations=DEBUG_ITERATIONS, backend="auto")
    logger.info("Running high-quality Richardson-Lucy...")
    result = deconvolve_volume(
        cropped,
        deconvolver,
        block_size=DEBUG_BLOCK_SIZE,
        overlap=psf.shape[0] // 2,
    )

    output_path = output_dir / f"{input_path.stem}_hq_debug.tif"
    logger.info("Saving output to %s", output_path)
    save_tiff(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
