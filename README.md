# specgrid

A Python library for preprocessing supernova spectra onto a fixed, logarithmically-spaced wavelength grid with constant velocity resolution.

## Overview

Observed spectra come with arbitrary wavelength ranges and varying resolutions. `snspectra` places every spectrum on the same master grid — from 3000 to 11000 Å — where each bin corresponds to a fixed velocity step `dv` in km/s:

```
Δlog₁₀(λ) = dv / c / ln(10)  →  constant resolution in velocity space
```

The full reduction pipeline applied to each spectrum is:

1. **`grid_spectrum`** — resamples the input (arbitrary resolution) onto the master log-spaced grid. Smoothing at two velocity scales (`dv` and `dvsmooth`) is used to estimate the uncertainty from the scatter between fine and coarse smoothing.
2. **Continuum division** — a broad smoothing (30 000 km/s) estimates the continuum. The spectrum is divided by it and shifted by −1, so absorption features appear as negative dips around zero.
3. **Apodization** — a cosine bell window is applied to the edges of the valid spectral region to suppress ringing from sharp boundaries.

## Installation

```bash
pip install git+https://github.com/jurados/spectra_normalized_grid.git
```

## Quick start

```python
import numpy as np
from snspectra import Spectra

# --- Synthetic supernova spectrum at arbitrary resolution ---
wave = np.linspace(4000, 9000, 1500)          # arbitrary wavelength range and sampling
continuum = 1.0 / (wave / 5000.0) ** 2
p_cygni = (
    0.8 * np.exp(-0.5 * ((wave - 6150) / 150) ** 2)   # emission
  - 0.4 * np.exp(-0.5 * ((wave - 5900) / 150) ** 2)   # absorption
)
flux  = continuum * (1 + p_cygni) + np.random.normal(0, 0.02, len(wave))
dflux = np.full_like(flux, 0.02)

# --- Reduce ---
processor = Spectra(dv=200, dvsmooth=2000)   # velocity resolution: 200 km/s
result    = processor.reduce_spectrum(
    np.array([wave, flux, dflux])            # accepts ndarray, DataFrame or Series
)

# --- Output ---
wave_grid  = result['wave'].iloc[0]    # master grid, same for every spectrum
flux_final = result['flux'].iloc[0]    # continuum-divided, apodized
mask       = result['mask'].iloc[0]    # True where the spectrum has coverage
```

## API

### `Spectra(dv=200, dvsmooth=2000)`

| Parameter  | Description |
|------------|-------------|
| `dv`       | Velocity resolution of the output grid in km/s. Determines the number of bins via `Δlog₁₀(λ) = dv / c / ln(10)`. |
| `dvsmooth` | Velocity scale for the coarse smoothing used to estimate uncertainties inside `grid_spectrum`. |

### `grid_spectrum(wave, flux, dflux=None)`

Resamples a single spectrum onto the master log-spaced grid.

| Argument | Description |
|----------|-------------|
| `wave`   | Rest-frame wavelength array (Å). |
| `flux`   | Flux array. |
| `dflux`  | Not used — uncertainty is estimated internally. |

Returns `(wave_grid, flux_grid, dflux_grid)`. Bins outside the input wavelength range are filled with `NaN`.

### `reduce_spectrum(spectra, restframe=True)`

Full reduction pipeline. Accepts:
- `np.ndarray` of shape `(2, N)` or `(3, N)` — `[wave, flux]` or `[wave, flux, dflux]`
- `pd.DataFrame` with array-valued `wave`/`flux`/`dflux` columns (one row per spectrum)
- `pd.DataFrame` with scalar columns (one row per wavelength bin, single spectrum)
- `pd.Series` (single spectrum record)

If `restframe=True` and a `redshift` column is present, the wavelength is de-redshifted before gridding.

Returns a `pd.DataFrame` with one row per spectrum and columns:

| Column       | Description |
|--------------|-------------|
| `wave`       | Master wavelength grid (Å) |
| `flux`       | Continuum-divided, apodized flux |
| `dflux`      | Propagated uncertainty |
| `flux_nnorm` | Flux after gridding, before continuum division |
| `flux_cont`  | Estimated continuum |
| `flux_org`   | Original input flux |
| `mask`       | Boolean array — `True` where the spectrum has valid coverage |

## Settings

Default wavelength range and bin count are defined in `snspectra/settings.py`:

```python
settings = {
    'min_wave': 3000,   # Å
    'max_wave': 11000,  # Å
}
```

The number of bins is not set manually — it is derived automatically from `dv` to guarantee constant velocity resolution across the full grid.
