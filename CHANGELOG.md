# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH
  - MAJOR: breaking changes incompatible with previous versions
  - MINOR: new features, backwards compatible
  - PATCH: bug fixes, backwards compatible

<!--
HOW TO ADD AN ENTRY
-------------------
When you start working on a new version, add a new block at the top (below this comment)
using the template below. Move entries from [Unreleased] to the new version block when
you tag a release.

Allowed section labels (use only what applies):
  ### Added      — new features
  ### Changed    — changes to existing functionality
  ### Deprecated — features that will be removed in a future version
  ### Removed    — features removed in this version
  ### Fixed      — bug fixes
  ### Security   — vulnerability fixes

VERSION TEMPLATE
----------------
## [X.Y.Z] - YYYY-MM-DD
### Added
- 

### Changed
- 

### Fixed
- 
-->

## [Unreleased]
<!-- Add upcoming changes here as you work. Move to a versioned block on release. -->

## [0.1.0] - 2026-05-04
### Added
- `Spectra` class with full reduction pipeline for astronomical spectra.
- `grid_spectrum`: resamples any input spectrum onto a fixed log-spaced master grid
  with constant velocity resolution (`dv` km/s) between 3000–11000 Å.
- `reduce_spectrum`: orchestrates the full pipeline — gridding, continuum division
  (AstroDASH convention: flux/continuum − 1), and cosine-bell apodization.
- `smooth_spectrum`: velocity-aware smoothing with moving average or Savitzky-Golay.
- `apodization`: cosine bell window applied to the edges of the valid spectral region.
- Flexible input handling: accepts `np.ndarray`, `pd.DataFrame`, and `pd.Series`.
- Automatic de-redshifting when `restframe=True` and a `redshift` column is present.
- `pyproject.toml` for installation via `pip install git+...`.
