# ============= IMPORT LIBRARIES =============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.interpolate import UnivariateSpline
from .settings import settings

# =============================================

class Spectra():
    """"
    Class for preprocessing spectra data for the SNeMPhyVAE model.
    """

    def __init__(self, dv=200, dvsmooth=2000):

        self.initial_settings = settings
        self.CSPEED           = 3e5
        self.dv               = dv
        self.dvsmooth         = dvsmooth

        self.spectra_nbins    = self._compute_nbins(dv)
        self.initial_settings['spectra_nbins'] = self.spectra_nbins

    def obtain_data(self):
        """
        Load and filter the spectra data to remove entries with all NaN flux values.

        Returns:
        --------
        data: '~pd.DataFrame'
            Filtered DataFrame containing only spectra with valid flux data.
        """
        data = self._load_data()
        mask = data.flux.apply(lambda x: np.all(np.isnan(x)))
        return data[~mask].copy().reset_index(drop=True)

    def _observed2restframe(self, spectrum):
        """
        Generates the wavelength grid and extracts non-zero flux indices.
        Also performs de-redshifting to the rest-frame.

        Parameters:
        -----------
        spectrum: `~pd.Series`
            Spectral record that must contain 'lambda_min_grid', 'lambda_max_grid',
            and 'flux' keys.

        Returns:
        --------
        flux: `~np.ndarray`
        wave_range: `~np.ndarray`
        mask: `~np.ndarray`
        """
        wave_range = np.logspace(
            np.log10(spectrum.lambda_min_grid),
            np.log10(spectrum.lambda_max_grid),
            spectrum.nlambda_grid
        )
        redshift   = np.nan_to_num(float(spectrum['redshift']))
        wave_range = wave_range / (1 + redshift)
        flux       = spectrum['flux'].copy()
        mask       = ~np.isnan(flux)
        return flux, wave_range, mask

    def normalize_spectrum(self, flux, spectra_dict):
        """
        Normalizes the spectrum using the minmax method.

        Parameters:
        -----------
        flux: '~np.ndarray'
        spectra_dict: dict

        Returns:
        --------
        flux_normalized: '~np.ndarray'
        """
        mask      = spectra_dict['mask']
        norm_flux = flux[mask]

        if norm_flux.size == 0:
            return np.zeros_like(flux)
        if np.all(norm_flux == 0):
            return flux

        spectra_dict['flux_normalized'][mask] = (
            (norm_flux - np.min(norm_flux)) / (np.max(norm_flux) - np.min(norm_flux))
        )
        return spectra_dict['flux_normalized']

    def continuum_fitting(self, flux, wave, spectra_dict, nknots=13):
        """
        Obtain the continuum fitting of a given spectrum using spline interpolation.

        Parameters:
        -----------
        flux: '~np.ndarray'
        wave: '~np.ndarray'
        spectra_dict: dict
        nknots: int

        Returns:
        --------
        flux_continuum: '~np.ndarray'
        """
        mask         = spectra_dict['mask']
        flux_working = spectra_dict['flux_smooth']
        wave_spline  = wave[mask]
        flux_spline  = flux_working[mask]

        indx       = np.linspace(0, len(wave_spline) - 1, nknots, dtype=int)
        wave_knots = wave_spline[indx]
        flux_knots = flux_spline[indx]

        spline      = UnivariateSpline(wave_knots, flux_knots, k=3, s=0)
        spline_point = spline(wave)

        spectra_dict['flux_spline']          = spline_point
        spectra_dict['flux_continuum'][mask] = flux[mask] / spline_point[mask]

        return spectra_dict['flux_continuum']

    def apodization(self, flux, dflux, spectra_dict, fraction=0.05):
        """
        Apply apodization to the spectrum using a cosine bell window at the edges.

        Parameters
        ----------
        flux: '~np.ndarray'
        dflux: '~np.ndarray'
        spectra_dict: dict
        fraction: float
            Fraction of the spectrum to taper (default 5%).

        Returns
        -------
        flux_apodized: '~np.ndarray'
        dflux_apodized: '~np.ndarray'
        """
        mask       = spectra_dict['mask']
        apod_flux  = flux[mask]
        apod_dflux = dflux[mask]

        if len(apod_flux) == 0:
            return flux, dflux

        n_apod = max(1, int(len(apod_flux) * fraction))
        window = np.ones(len(apod_flux))
        x      = np.linspace(0, np.pi / 2, n_apod)
        window[:n_apod]  = np.sin(x) ** 2
        window[-n_apod:] = np.sin(x[::-1]) ** 2

        spectra_dict['flux_apodized'][mask]  = apod_flux  * window
        spectra_dict['dflux_apodized'][mask] = apod_dflux * window

        return spectra_dict['flux_apodized'], spectra_dict['dflux_apodized']

    def grid_spectrum(self, wave, flux, dflux=None):
        """
        Resample spectrum onto the master log-spaced grid at constant velocity resolution.
        Uncertainty is estimated from the scatter between fine and coarse smoothing.

        Parameters:
        ------------
        wave: '~np.ndarray'
            Input wavelength array (rest-frame).
        flux: '~np.ndarray'
            Input flux array.
        dflux: '~np.ndarray', optional
            Not used — uncertainty is estimated internally from flux_fine - flux_coarse.

        Returns:
        --------
        wave_grid: '~np.ndarray'
            Master wavelength grid [min_wave, max_wave] with constant velocity spacing.
        flux_grid: '~np.ndarray'
            Flux resampled to master grid (NaN outside spectrum coverage).
        dflux_grid: '~np.ndarray'
            Estimated uncertainty resampled to master grid.
        """
        wave_min  = self.initial_settings['min_wave']
        wave_max  = self.initial_settings['max_wave']
        wave_grid = np.logspace(np.log10(wave_min), np.log10(wave_max), self.spectra_nbins, endpoint=True)

        # Convert velocity widths to log10(λ) windows expressed as Timedelta seconds
        # so that pandas rolling() operates in log-wavelength space (constant in velocity).
        dlog10wave       = self.dv       / self.CSPEED / np.log(10) * 86400
        dlog10wavesmooth = self.dvsmooth / self.CSPEED / np.log(10) * 86400

        sn = pd.DataFrame({'wave': wave, 'flux': flux})
        sn['log10wave_idx'] = sn['wave'].apply(lambda x: pd.Timedelta(np.log10(x), 'days'))
        sn = sn.set_index('log10wave_idx').sort_index()

        flux_fine   = sn['flux'].rolling(f'{int(dlog10wave)}s',      center=True).mean()
        flux_coarse = sn['flux'].rolling(f'{int(dlog10wavesmooth)}s', center=True).mean()
        dflux_est   = (flux_fine - flux_coarse).rolling(f'{int(dlog10wavesmooth)}s', center=True).std()

        valid      = flux_fine.notna()
        orig_wave  = sn.loc[valid, 'wave'].values
        orig_flux  = flux_fine[valid].values
        orig_dflux = dflux_est[valid].values

        flux_grid  = np.interp(wave_grid, orig_wave, orig_flux,  left=np.nan, right=np.nan)
        dflux_grid = np.interp(wave_grid, orig_wave, orig_dflux, left=np.nan, right=np.nan)

        return wave_grid, flux_grid, dflux_grid

    def reduce_spectrum(self, spectra, restframe=True):
        """
        Full reduction pipeline for a set of spectra:
          1. grid_spectrum  — resample to master log-spaced grid (constant velocity resolution)
          2. Continuum division (coarse smoothing)
          3. Apodization

        Parameters:
        -----------
        spectra: np.ndarray | pd.DataFrame | pd.Series
            Input spectra. Accepted formats:
            - np.ndarray of shape (2, N) [wave, flux] or (3, N) [wave, flux, dflux]
            - pd.DataFrame with scalar 'wave'/'flux' columns (one spectrum per row of pixels)
            - pd.DataFrame where each row is one spectrum and 'flux'/'wave' are arrays
            - pd.Series (single spectrum record)
        restframe: bool
            If True, de-redshift using the 'redshift' column before gridding.

        Returns:
        --------
        result_df: '~pd.DataFrame'
            DataFrame with one row per spectrum and columns:
            'wave', 'flux', 'dflux', 'flux_nnorm', 'flux_cont', 'flux_org', 'dflux_org', 'mask'.
        """
        results = []

        wave_min_grid = self.initial_settings['min_wave']
        wave_max_grid = self.initial_settings['max_wave']

        # --- INPUT NORMALIZATION ---
        if isinstance(spectra, np.ndarray):
            if spectra.shape[0] == 2:
                wave, flux = spectra
                dflux = np.zeros_like(flux)
            elif spectra.shape[0] >= 3:
                wave, flux, dflux = spectra[:3]
            else:
                raise ValueError("NumPy array must have at least 2 rows (wave, flux)")
            spectra = pd.DataFrame([{
                'oid':              'unknown',
                'snname':           'unknown',
                'mjd':              np.nan,
                'wave':             wave,
                'flux':             flux,
                'dflux':            dflux,
                'lambda_min':       np.min(wave),
                'lambda_max':       np.max(wave),
                'lambda_min_grid':  wave_min_grid,
                'lambda_max_grid':  wave_max_grid,
                'redshift':         0.0,
                'type':             'Unknown',
            }])

        elif isinstance(spectra, pd.DataFrame):
            if 'flux' in spectra.columns and np.isscalar(spectra['flux'].iloc[0]):
                wave  = spectra['wave'].values if 'wave' in spectra.columns else np.arange(len(spectra))
                flux  = spectra['flux'].values
                dflux = spectra['dflux'].values if 'dflux' in spectra.columns else np.zeros_like(flux)
                spectra = pd.DataFrame([{
                    'oid':              'unknown',
                    'snname':           'unknown',
                    'mjd':              np.nan,
                    'wave':             wave,
                    'flux':             flux,
                    'dflux':            dflux,
                    'lambda_min':       np.min(wave),
                    'lambda_max':       np.max(wave),
                    'lambda_min_grid':  wave_min_grid,
                    'lambda_max_grid':  wave_max_grid,
                    'redshift':         0.0,
                    'type':             'Unknown',
                }])

        if isinstance(spectra, pd.Series):
            spectra = pd.DataFrame([spectra])

        for _, spectrum in tqdm(spectra.iterrows(), total=len(spectra), desc="Reducing spectra"):

            wave  = spectrum['wave'].copy()
            flux  = spectrum['flux'].copy()
            dflux = spectrum['dflux'].copy()

            # De-redshift to rest-frame
            if restframe:
                redshift = np.nan_to_num(float(spectrum.get('redshift', 0.0)))
                wave = wave / (1 + redshift)

            dflux = np.nan_to_num(
                dflux,
                nan=np.nanmean(dflux) if not np.all(np.isnan(dflux)) else 1e-20
            )

            # 1. Resample to master log-spaced grid at constant velocity resolution
            wave_grid, flux_grid, dflux_grid = self.grid_spectrum(wave, flux, dflux)

            mask = ~np.isnan(flux_grid)

            # Shift to non-negative and avoid exact zeros
            if mask.any():
                min_val = np.nanmin(flux_grid[mask])
                if min_val < 0:
                    flux_grid[mask] += np.abs(min_val)
                flux_grid[mask] += 1e-20

            flux_grid  = np.nan_to_num(flux_grid,  nan=0.0)
            dflux_grid = np.nan_to_num(dflux_grid, nan=0.0)

            # 2. Continuum via coarse smoothing
            flux_continuum = np.zeros_like(flux_grid)
            valid_idx = np.flatnonzero(flux_grid)
            if len(valid_idx) > 0:
                start, end = valid_idx[0], valid_idx[-1] + 1
                flux_continuum[start:end] = self.smooth_spectrum(
                    flux_grid[start:end],
                    wave_grid[start:end],
                    method='moving_average',
                    velocity=30000,
                )
            flux_continuum[flux_continuum == 0] = 1e-20

            # 3. Divide by continuum (AstroDASH convention: subtract 1)
            flux_cont  = flux_grid  / flux_continuum - 1
            dflux_cont = dflux_grid / flux_continuum

            spectra_dict = {
                'mask':           mask,
                'flux_apodized':  np.zeros_like(wave_grid),
                'dflux_apodized': np.zeros_like(wave_grid),
            }

            # 4. Apodization
            final_flux, final_dflux = self.apodization(flux_cont, dflux_cont, spectra_dict)

            results.append({
                'oid':       spectrum['oid'],
                'mjd':       spectrum['mjd'],
                'redshift':  spectrum.get('redshift', np.nan),
                'wave':      wave_grid,
                'flux':      final_flux,
                'dflux':     final_dflux,
                'flux_nnorm': flux_grid,
                'flux_cont':  flux_continuum,
                'flux_org':   spectrum['flux'].copy(),
                'dflux_org':  spectrum['dflux'].copy(),
                'mask':       mask,
            })

        return pd.DataFrame(results)

    def _compute_nbins(self, dv):
        """
        Compute the number of bins for a given velocity resolution.

        Parameters:
        -----------
        dv: float
            Velocity resolution in km/s.

        Returns:
        --------
        nbins: int
        """
        wave_min = self.initial_settings['min_wave']
        wave_max = self.initial_settings['max_wave']
        step_log = dv / self.CSPEED / np.log(10)
        return int(np.ceil((np.log10(wave_max) - np.log10(wave_min)) / step_log)) + 1

    def slice_wavelength(self, wave, target_size=None):
        """
        Reduce the size of a wavelength array using log-spaced interpolation.

        Parameters:
        ------------
        wave: '~np.ndarray'
        target_size: int or None

        Returns:
        --------
        wave_reduced: np.ndarray
        """
        if target_size is None:
            target_size = self.initial_settings['spectrum_bins']
        if target_size == len(wave):
            return wave
        return np.logspace(np.log10(wave.min()), np.log10(wave.max()), target_size)

    def smooth_spectrum(self, spectrum, wave, method='moving_average', velocity=10000, polyorder=2, target_size=None):
        """
        Smooth a spectrum using a window with constant width in velocity space.

        Parameters:
        ------------
        spectrum: '~np.ndarray'
        wave: '~np.ndarray'
        method: str ('savgol' or 'moving_average')
        velocity: float
            Smoothing window in km/s.
        polyorder: int
            Polynomial order for Savitzky-Golay filter.
        target_size: int or None
            If given, resample the smoothed spectrum to this size via log-interpolation.

        Returns:
        --------
        flux_smoothed: np.ndarray
        """
        from scipy.signal import savgol_filter

        delta_lambda         = velocity / self.CSPEED * wave[len(wave) // 2]
        average_delta_lambda = np.median(np.diff(wave))
        window_size          = int(delta_lambda // average_delta_lambda)
        window_size          = max(3, int(min(window_size, len(spectrum) - 1)))
        window_size          = window_size if window_size % 2 != 0 else window_size + 1

        if method == 'savgol':
            if polyorder >= window_size:
                polyorder = max(1, window_size - 1)
            flux_smoothed = savgol_filter(spectrum, window_length=window_size, polyorder=polyorder, mode='interp')
        elif method == 'moving_average':
            kernel        = np.ones(window_size) / window_size
            flux_smoothed = np.convolve(spectrum, kernel, mode='same')

        if target_size is None:
            return flux_smoothed

        log_wave_tgt  = np.linspace(np.log10(wave.min()), np.log10(wave.max()), target_size)
        flux_smoothed = np.interp(log_wave_tgt, np.log10(wave), flux_smoothed)
        return flux_smoothed


if __name__ == "__main__":

    wave_min = np.random.uniform(3000, 4000)
    wave_max = np.random.uniform(8000, 10000)
    wave     = np.linspace(wave_min, wave_max, 1000)

    continuum = 1.0 / (wave / 5000.0) ** 2
    p_cygni   = (
        0.8 * np.exp(-0.5 * ((wave - 6150) / 150) ** 2)
        - 0.4 * np.exp(-0.5 * ((wave - 5900) / 150) ** 2)
    )
    flux = continuum * (1 + p_cygni) + np.random.normal(0, 0.05, size=len(wave))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave, flux, alpha=0.9)
    ax.set_ylabel('Flux')
    ax.set_xlabel('Wavelength [Å]')
    plt.tight_layout()
    plt.show()

    spectrum = pd.DataFrame({
        'wave':  wave,
        'flux':  flux,
        'dflux': np.random.normal(0, 0.01, size=len(wave)),
    })

    processor = Spectra()
    result    = processor.reduce_spectrum(spectrum)
    print(result)
