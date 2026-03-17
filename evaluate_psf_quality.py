"""Evaluate PSF/reconstruction quality metrics for microscopy volumes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pydeconlab.io import _normalize_psf, load_tiff

logger = logging.getLogger(__name__)

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency
    structural_similarity = None


def _fwhm_from_profile(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 3:
        return float("nan")

    peak_idx = int(np.argmax(profile))
    peak_value = float(profile[peak_idx])
    if peak_value <= 0:
        return float("nan")

    half_max = peak_value * 0.5

    left = peak_idx
    while left > 0 and profile[left] >= half_max:
        left -= 1
    right = peak_idx
    while right < profile.size - 1 and profile[right] >= half_max:
        right += 1

    if left == peak_idx or right == peak_idx:
        return float("nan")

    left_x = float(left)
    if profile[left] != profile[left + 1]:
        left_x += (half_max - profile[left]) / (profile[left + 1] - profile[left])

    right_x = float(right)
    if profile[right] != profile[right - 1]:
        right_x -= (half_max - profile[right]) / (profile[right - 1] - profile[right])

    return max(right_x - left_x, 0.0)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2, dtype=np.float64))
    if mse <= 0:
        return float("inf")
    data_range = float(reference.max() - reference.min())
    if data_range <= 0:
        data_range = 1.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    if structural_similarity is not None:
        ref = np.asarray(reference, dtype=np.float32)
        cand = np.asarray(candidate, dtype=np.float32)
        data_range = float(max(ref.max(), cand.max()) - min(ref.min(), cand.min()))
        data_range = data_range if data_range > 0 else 1.0
        scores = []
        for z in range(ref.shape[0]):
            scores.append(
                structural_similarity(ref[z], cand[z], data_range=data_range)
            )
        return float(np.mean(scores))

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    data_range = float(max(ref.max(), cand.max()) - min(ref.min(), cand.min()))
    data_range = data_range if data_range > 0 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(ref.mean())
    mu_y = float(cand.mean())
    sigma_x = float(ref.var())
    sigma_y = float(cand.var())
    sigma_xy = float(((ref - mu_x) * (cand - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


def bead_fwhm(psf: np.ndarray) -> tuple[float, float]:
    normalized = _normalize_psf(psf)
    peak = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    z, y, x = peak
    profile_z = normalized[:, y, x]
    profile_y = normalized[z, :, x]
    profile_x = normalized[z, y, :]
    fwhm_xy = float(np.nanmean([_fwhm_from_profile(profile_x), _fwhm_from_profile(profile_y)]))
    fwhm_z = float(_fwhm_from_profile(profile_z))
    return fwhm_xy, fwhm_z


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PSF and reconstruction quality.")
    parser.add_argument("--candidate", required=True, help="Candidate TIFF stack")
    parser.add_argument("--reference", default=None, help="Optional reference TIFF stack")
    parser.add_argument("--psf", default=None, help="Optional PSF TIFF for FWHM evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    candidate = load_tiff(Path(args.candidate), memmap=False).astype(np.float32, copy=False)

    if args.psf:
        psf = load_tiff(Path(args.psf), memmap=False).astype(np.float32, copy=False)
        fwhm_xy, fwhm_z = bead_fwhm(psf)
        logger.info("PSF FWHM XY: %.3f px", fwhm_xy)
        logger.info("PSF FWHM Z : %.3f px", fwhm_z)

    if args.reference:
        reference = load_tiff(Path(args.reference), memmap=False).astype(np.float32, copy=False)
        if reference.shape != candidate.shape:
            raise ValueError("Reference and candidate volumes must have the same shape.")
        logger.info("PSNR: %.3f dB", psnr(reference, candidate))
        logger.info("SSIM: %.5f", ssim(reference, candidate))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
