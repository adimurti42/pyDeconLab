# pyDeconLab

A small, plugin-based Python package implementing common 3D microscopy deconvolution algorithms:
Richardson-Lucy, Richardson-Lucy+TV, Landweber, and Tikhonov-Miller.

Install:
    pip install numpy scipy tifffile pytest

Requirements:
    numpy, scipy, tifffile, scikit-image, pyfftw, pyotf

Run CLI example:
    python -m pydeconlab.main --algo RL --input my_stack.tif --psf my_psf.tif --iterations 30 --output deconv.tif

Run tests:
    pytest -q

Notes:
 - The code uses scipy.signal.fftconvolve for efficient FFT-based convolution on CPU.
 - Backend toggle 'cupy' is present, but production GPU usage requires proper cupy and cupyx installations and was not fully implemented here. The code falls back to NumPy/scipy if cupy is not available.
 - PSF generation: pass a .tif or a JSON config specifying {'type':'gaussian','shape':[z,y,x],'sigma':..}
 - Optional pyotf support: if a PSF generation dict requests type 'pyotf' or 'gibson-lanni' and pyotf is installed, pyotf will be used. Otherwise we fallback to a Gaussian PSF.
