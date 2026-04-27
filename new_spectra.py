# ============= IMPORT LIBRARIES =============
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from scipy.interpolate import UnivariateSpline

# Import custom settings
#if os.getcwd().endswith('notebooks'):
#    PROJECT_ROOT = os.path.dirname(os.getcwd())
#    from SNeMPhyVAE.model.settings import initial_settings, band_info
#    from SNeMPhyVAE.model.lightcurves import LightCurves
#else:
#    PROJECT_ROOT = os.getcwd()
#    from settings import initial_settings, band_info
#    from lightcurves import LightCurves

# =============================================

class Spectra():
    """"
    Class for preprocessing spectra data for the SNeMPhyVAE model.
    """

    def __init__(self):#, settings=initial_settings):

        self.initial_settings = settings

    def obtain_data(self):
        """
        Load and filter the spectra data to remove entries with all NaN flux values.
        
        Returns:
        --------
        data: '~pd.DataFrame'
            Filtered DataFrame containing only spectra with valid flux data.
        """
        data = self._load_data()
        
        # This mask removes spectra with all NaN flux values
        mask = data.flux.apply(lambda x: np.all(np.isnan(x)))

        return data[~mask].copy().reset_index(drop=True)

    def _observed2restframe(self, spectrum, rest_frame=True, redshift=None):
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
        flux_nonzero: `~np.ndarray`
            Array containing flux values at non-zero indices.
        wave_nonzero: `~np.ndarray`
            Array containing wavelengths corresponding to non-zero indices.
        mask: `~np.ndarray`
            Boolean mask indicating non-NaN flux values.
        """
        # Create wavelength grid and extract flux
        wave_range = np.logspace(
            np.log10(spectrum.lambda_min_grid),
            np.log10(spectrum.lambda_max_grid),
            spectrum.nlambda_grid
        )
        
        # De-redshifting (To rest-frame)
        redshift   = np.nan_to_num(float(spectrum['redshift']))
        wave_range = wave_range / (1 + redshift)

        flux = spectrum['flux'].copy()

        # Get mask for non-NaN flux values
        mask = ~np.isnan(flux)

        return flux, wave_range, mask

    def normalize_spectrum(self, flux, spectra_dict):
        """
        Normalizes the spectrum using the minmax method.

        Parameters:
        -----------
        spectrum: '~pd.Series'
            Spectral record.

        Returns:
        --------
        flux_normalized: '~np.ndarray'
            Normalized flux.
        """
        mask = spectra_dict['mask']
        norm_flux = flux[mask]
        
        if norm_flux.size == 0:
            print(f"Warning: Empty flux array for spectrum")
            return np.zeros_like(flux)

        if np.all(norm_flux == 0):
            return flux

        spectra_dict['flux_normalized'][mask] = (norm_flux - np.min(norm_flux)) / (np.max(norm_flux) - np.min(norm_flux))
        return spectra_dict['flux_normalized']

    def continuum_fitting(self, flux, wave, spectra_dict, nknots=13) -> np.ndarray:
        """
        Obtain the continuum fitting of a given spectrum using spline interpolation.
        
        Parameters:
        -----------
        flux: '~np.ndarray'
            specum flux (normalized).
        wave: '~np.ndarray'
            wavelengths.
        spectra_dict: dict
            Dictionary with intermediate information obtained from _grid_flux.

        Returns:
        --------
        spectrum_flux '~np.ndarray'
            Spectrum with the continuum fitting applied (flux divided by the continuum).
        """

        mask = spectra_dict['mask']
        flux_working = spectra_dict['flux_smooth']

        #wave_spline = wave.copy()
        #flux_spline = flux_working.copy()
        wave_spline = wave[mask]
        flux_spline = flux_working[mask]
        
        #if len(wave_spline) < 13:
        #    print('There are not enought data. < 13')
        #    return flux
        
        indx = np.linspace(0, len(wave_spline)-1, nknots, dtype=int)
        
        wave_knots = wave_spline[indx] 
        flux_knots = flux_spline[indx]
        
        # Fit a spline to the knots
        # If k=3 (cubic spline) is too oscillatory.
        spline = UnivariateSpline(wave_knots, flux_knots, k=3, s=0) 
        # This is based on AstroDASH tutorial remake the spline
        
        #spline_wave  = np.linspace(wave_spline.min(), wave_spline.max(), nknots)
        #spline_point = spline(spline_wave)
        #spline       = UnivariateSpline(spline_wave, spline_point, k=3)
        #spline_point = spline(wave_spline)
        spline_point  = spline(wave) 
        spectra_dict['flux_spline'] = spline_point
        #spectra_dict['flux_spline'][mask] = spline_point
        
        # Save the continuum flux in the spectra_dict dictionary
        spectra_dict['flux_continuum'][mask] = flux[mask] / spline_point[mask]
        
        #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        #ax[0].plot(wave, flux, label='Processed Flux', color='blue', alpha=0.5)
        #ax[0].plot(wave, spectra_dict['flux_spline'], label='Continuum Spline', color='red', alpha=0.7)
        #ax[1].plot(wave, spectra_dict['flux_continuum'], label='Flux / Continuum', color='green', alpha=0.7)
        #ax[0].scatter(wave_knots, flux_knots, label='Spline Control Points', color='black', zorder=5) # zorder=5 los pone al frente
        #ax[0].set_ylim(bottom=0)
        #ax[0].set_ylabel('Flux')
        #ax[1].set_xlabel('Wavelength [Angstrom]') 
        #ax[0].legend(frameon=False)
        #fig.suptitle('Continuum Fitting', fontsize=16)
        #plt.show()

        return spectra_dict['flux_continuum']

    def apodization(self, flux, dflux, spectra_dict, fraction=0.05):
        """
        Apply apodization to the spectrum using a 'cosine bell' window 
        at the start and end.

        Parameters
        ----------
        flux: '~np.ndarray'
            Flux vector.
        dflux: '~np.ndarray'
            Uncertainty in the flux vector.
        spectra_dict: dict
            Dictionary with intermediate information obtained from _grid_flux.
        fraction: float
            Spectrum fraction to apply the window (default 5%).

        Returns
        -------
        flux_apodized: '~np.ndarray'
            Apodized flux vector.
        """
        mask       = spectra_dict['mask']
        apod_flux  = flux[mask]
        apod_dflux = dflux[mask]

        # There is not enought data to apply the apodization.        
        if len(apod_flux) == 0: return flux, dflux

        n_apod = max(1, int(len(apod_flux) * fraction))

        window = np.ones(len(apod_flux))
        x      = np.linspace(0, np.pi/2, n_apod)
        window[:n_apod]  = np.sin(x)**2
        window[-n_apod:] = np.sin(x[::-1])**2

        spectra_dict['flux_apodized'][mask]  = apod_flux * window
        spectra_dict['dflux_apodized'][mask] = apod_dflux * window
        
        return spectra_dict['flux_apodized'], spectra_dict['dflux_apodized']

    def preprocess_spectrum(self, spectra, smooth_spectrum=False):
        """
        Process a spectra set and add columns with the processed information.

        Parameters:
        -----------
        spectra : '~pd.DataFrame'
            DataFrame containing spectra records or a single spectrum as a pd.Series.
        smooth_spectrum : bool
            If True, the final spectrum will be sliced to a target size using smoothing and 
            interpolation.

        Returns:
        --------
        new_spectra : '~pd.DataFrame'
            DataFrame with the processed spectra, including columns like:
            'flux', 'wave', 'flux_continuum', 'flux_normalized', and 'final_spectrum'.
        """

        results = []

        if isinstance(spectra, pd.Series): spectra = pd.DataFrame([spectra])

        # Iterarate over each spectrum
        for i, (_, spectrum) in enumerate(tqdm(spectra.iterrows(), total=len(spectra), desc="Processing spectra")):

            # Original data
            flux_org  = spectrum['flux'].copy()
            dflux_org = spectrum['dflux'].copy()

            wave_range = np.logspace(
                np.log10(spectrum.lambda_min_grid),
                np.log10(spectrum.lambda_max_grid),
                spectrum.nlambda_grid
            )

            wave  = wave_range
            flux  = flux_org
            dflux = dflux_org
            dflux = np.nan_to_num(dflux, nan=np.nanmean(dflux) if not np.all(np.isnan(dflux)) else 1e-20)  # Replace NaN with mean or small value
            
            mask = ~np.isnan(flux) # This mask removes NaN flux values
            flux = np.nan_to_num(flux)
            flux[mask] += np.abs(np.min(flux[mask])) # Avoid negative flux values
            flux[mask] = flux[mask] + 1e-20          # Avoid zero values
            
            flux_smooth = self.smooth_spectrum(
                flux,
                wave,
                method='moving_average',
                velocity=1,
                target_size=self.initial_settings['spectrum_bins'],
            )
            
            #print('Flux after smoothing:', len(flux_smooth))
            
            
            #print(flux_smooth)
        
            wave_smooth = self.slice_wavelength(wave, target_size=self.initial_settings['spectrum_bins'])
            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=True)
            #ax.plot(wave, flux, label='Original Flux', color='orange', alpha=0.5)
            #ax.plot(wave_smooth, flux_smooth, label='Smoothed Flux', color ='blue', alpha=0.5)
            #ax.set_ylabel('Flux')
            #ax.legend(frameon=False)
            #fig.suptitle(f'Spectrum OID: {spectrum.oid} - InitialSmoothing', fontsize=16)    
            #plt.show()

            log_wave_org = np.log10(wave)
            log_wave_new = np.log10(wave_smooth)
            dflux_smooth = np.interp(log_wave_new, log_wave_org, dflux)

            mask = flux_smooth != 0.0            
            flux = flux_smooth.copy()
            wave = wave_smooth.copy()

            valid_idx  = np.flatnonzero(flux)
            start, end = valid_idx[0], valid_idx[-1] + 1
            flux_continuum = np.zeros_like(wave)
            continuum_values = self.smooth_spectrum(
                flux[start:end],
                wave[start:end],
                method='moving_average',
                velocity=30000,
                target_size = None
            )
            flux_continuum[start:end] = continuum_values
            
            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=True)
            #ax.plot(wave, flux, label='Original Flux', color='orange', alpha=0.5)
            #ax.set_ylabel('Flux')
            #ax.legend(frameon=False)
            #ax.plot(wave, flux_smooth, label='          Processed Flux', color ='blue', alpha=0.5)  
            #fig.suptitle(f'Spectrum OID: {spectrum.oid} - InitialSmoothing', fontsize=16)    
            #plt.show()
            
            spectra_dict = {
                'oid':             spectrum.oid,
                'wave_range':      wave,
                'flux':            flux,
                #'flux_smooth':     flux_smooth,
                #'flux_continuum':  np.zeros_like(wave),
                'flux_apodized':   np.zeros_like(wave),
                'dflux_apodized':  np.zeros_like(wave),
                #'flux_normalized': np.zeros_like(wave),
                'mask':            mask,
            }
                
            #continuum_spectra = self.continuum_fitting(flux, wave, spectra_dict)
            flux_continuum[flux_continuum == 0] = 1e-20  # Avoid division by zero
            continuum_spectra = flux / flux_continuum
            dflux_continuum   = dflux_smooth / flux_continuum

            # This is made by the AstroDASH tutorial
            continuum_spectra = continuum_spectra - 1
            
            #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
            #ax[0].plot(wave, flux, label='Original Flux', color='orange', alpha=0.5)
            #ax[0].set_ylabel('Flux')
            #ax[0].legend(frameon=False)
            #ax[0].plot(wave, flux_smooth, label='Processed Flux', color='blue', alpha=0.5)
            #
            #ax[1].plot(wave, continuum_spectra, label='Continuum Flux', color='green', alpha=0.5)
            #ax[1].legend(frameon=False)
            #ax[1].set_ylabel('Flux Continuum')
            #ax[1].set_xlabel('Wavelength [Angstrom]')
            #fig.suptitle(f'Spectrum OID: {spectrum.oid}', fontsize=16)
            #plt.show()
            
            #norm_result = self.normalize_spectrum(continuum_result, spectra_dict)

            final_flux, final_dflux = self.apodization(continuum_spectra, dflux_continuum, spectra_dict)

            #final_spectra = self.normalize_spectrum(spec_flux, spectra_dict)
            #if len(spec_flux) == 0:
            #    continue

            #np.clip(spec_flux, a_min=0.0, a_max=None, out=spec_flux)
            #mask = spec_flux > 0

            # Actualizar los arrays en el estado en las posiciones de índices no nulos
            #mask = spectra_dict['nonzero_mask']
            #spectra_dict['flux_normalized'][mask] = norm_flux
            #spectra_dict['continuum_flux'][mask]  = spectra_dict['flux_continuum_fit']

            #spectra_dict['final_spectrum'][mask]  = spec_flux
            spectra_dict['final_flux']  = final_flux
            spectra_dict['final_dflux'] = final_dflux

            #print(type(spec_flux))
            #final_spectrum[idx]  = self.smooth_spectrum(spec_flux, target_size=self.initial_settings['spectrum_bins'])

            # Crear un diccionario con los resultados; se usa la grilla completa
            results.append({
                'oid':             spectrum.oid,
                'snname':          spectrum.snname,
                'mjd':             spectrum.mjd,
                'redshift':        spectrum.get('redshift', np.nan),
                'type':            spectrum.get('type', 'Unknown'),
                'wave':            spectra_dict['wave_range'],
                #'flux_spline':     spectra_dict['flux_spline'],
                #'flux_apodized':   spectra_dict['flux_apodized'],
                #'flux_normalized': spectra_dict['flux_normalized'],
                'flux':            spectra_dict['final_flux'],
                'dflux':           spectra_dict['final_dflux'],
                'flux_nnorm':      flux,
                'flux_cont':       flux_continuum,
                'flux_org':        flux_org,
                'dflux_org':       dflux_org,
                'mask':            spectra_dict['mask'],
                'lambda_min':      spectrum.lambda_min,
                'lambda_max':      spectrum.lambda_max,
                'lambda_min_grid': spectrum.lambda_min_grid,
                'lambda_max_grid': spectrum.lambda_max_grid,
                'nlambda_grid':    spectrum.nlambda_grid,
                'spectrum_bins':   self.initial_settings['spectrum_bins'],
            })
            result_df = pd.DataFrame(results)
        
        return result_df
    
    def slice_wavelength(self, wave, target_size=None):
        """
        Reduce the size of a wavelength array using interpolation.

        Parameters:
        ------------
        - wave: '~np.ndarray': 
            Input wavelength array.
        - target_size: int or None
            Desired output size. If None, no size reduction is applied.
        Returns:
        --------
            np.ndarray: Wavelength array reduced to target_size.
        """
        if target_size is None: 
            target_size = self.initial_settings['spectrum_bins']

        if target_size == len(wave):
            return wave

        wave_reduced = np.logspace(np.log10(wave.min()), np.log10(wave.max()), target_size)
        return wave_reduced

    def smooth_spectrum(self, spectrum, wave, method='moving_average', velocity=10000, polyorder=2, target_size=None):
        """
        Smooth and reduce the size of a spectrum using interpolation or averaging.

        Parameters:
        ------------
        - spectrum: '~np.ndarray': 
            Input spectrum.
        - method: str 
            Smoothing method ('savgol', 'gaussian', 'moving_average').
        - window_size: int 
            Size of the smoothing window (must be odd).
        - polyorder: int 
            Polynomial order for Savitzky-Golay filter (if used).
        - sigma: float 
            Standard deviation for Gaussian filter (if used).
        - target_size: int or None
            Desired output size. If None, no size reduction is applied.
        Returns:
        --------
            np.ndarray: Espectro suavizado y reducido.
        """
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import savgol_filter

        CSPEED = 3e5  # Speed of light in km/s
        delta_lambda = velocity / CSPEED * wave[len(wave)//2]  # Approximate delta_lambda for the given velocity
        average_delta_lambda = np.median(np.diff(wave))
        
        window_size = int(delta_lambda // average_delta_lambda)
        window_size = max(3, int(min(window_size, len(spectrum)-1)))
        window_size = window_size if window_size % 2 != 0 else window_size + 1

        # Apply smoothing
        if method == 'savgol':
            if polyorder >= window_size:
                polyorder = max(1, window_size - 1)
            flux_smoothed = savgol_filter(spectrum, window_length=window_size, polyorder=polyorder, mode='interp')

        elif method == 'moving_average':
            kernel   = np.ones(window_size) / window_size
            flux_smoothed = np.convolve(spectrum, kernel, mode='same')
            #flux_smoothed = uniform_filter1d(spectrum, size=window_size, mode='nearest')

        if target_size is None:
            return flux_smoothed
        
        # Size reduction if target_size is specified
        if target_size is not None and target_size != len(flux_smoothed):
            log_wave_min  = np.log10(wave.min())
            log_wave_max  = np.log10(wave.max())
            log_wave_tgt  = np.linspace(log_wave_min,log_wave_max,target_size)
            flux_smoothed = np.interp(log_wave_tgt, np.log10(wave), flux_smoothed)
            return flux_smoothed

if __name__ == "__main__":

    # Generar espectro sintetico de supernova
    wave_min = np.random.uniform(3000, 4000)
    wave_max = np.random.uniform(8000, 10000)
    wave = np.linspace(wave_min, wave_max, 1000)
    
    # Continuo simple y perfil P-Cygni (emision + absorcion desplazada al azul) + ruido
    continuum = 1.0 / (wave / 5000.0)**2
    p_cygni = 0.8 * np.exp(-0.5 * ((wave - 6150) / 150)**2) - 0.4 * np.exp(-0.5 * ((wave - 5900) / 150)**2)
    flux = continuum * (1 + p_cygni) + np.random.normal(0, 0.05, size=len(wave))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=True)
    ax.plot(wave, flux, alpha=0.9)
    ax.set_ylabel('Final Processed Flux')
    plt.show()


    #spectra_processor = Spectra()
    #spectra = spectra_processor.obtain_data()
    #spectra = spectra.iloc[:10]  # For testing, limit to first 100 spectra
    #print(f"Total spectra: {len(spectra)}")
    #proc_spec = spectra_processor.preprocess_spectrum(spectra, smooth_spectrum=True)
    #print("Columns:", proc_spec.columns)
    #print("Spectra shape:", proc_spec.shape)
    #print("Number of unique objects:", proc_spec['oid'].nunique())
    #proc_spec.to_pickle('../SNeMPhyVAE/data/preprocessed_spectra_snia.pkl')