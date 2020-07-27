from astropy.io import fits
import numpy as np

def reconstruct_lsf(wavelengths, resolution_file, ext=1, reconstruct_full=False):
    """
    Reconstruct the median line spread function (LSF) for MAGPI data as a function of wavelength. 
    Optionally can be used to reconstruct the full spatially-variying LSF.
    
    Returns the second moment of the LSF in Angtroms.

    Parameters
    ----------
    wavelengths: float, array
        wavelengths (in Angstroms) at which the resolution should be
        reconstructed.

    resolution_file: string
        Name of the file containing the relevant lsf extension

    ext: int
        Extension of resolution_file that contains the lsf information

    reconstruct_full: boolean
        If True, reconstruct the LSF at every spatial position across the field.  Otherwise
        just return the spatially-averaged LSF. (default: False)

    """
    
    #make sure that single values are made to conform to the array requirements
    wavelengths = np.atleast_1d(wavelengths)
    
    #pull wavelength information
    res_header = fits.getheader(resolution_file, ext=ext)
    wave_range = np.array([res_header['MAGPI LSF WMIN'], res_header['MAGPI LSF WMAX']])

    #compute normalized wavelengths for reconstruction
    norm_wavelength = (wavelengths - wave_range.mean())*2/wave_range.ptp()
    
    wstack = np.vstack((norm_wavelength**2, norm_wavelength, np.ones_like(norm_wavelength)))
    if reconstruct_full: #generate the full LSF
        poly_coeffs = fits.getdata(resolution_file, ext=ext)
        
        poly_out = 0.
        for ii in range(3):
            poly_out += poly_coeffs[ii][np.newaxis,:,:]*wstack[ii,:][:,np.newaxis,np.newaxis]
        return poly_out
    
    else: #just reconstructing the average field LSF
        poly_coeffs = np.array([res_header['MAGPI LSF COEFF0'],
                                res_header['MAGPI LSF COEFF1'],
                                res_header['MAGPI LSF COEFF2']]
                              )
        
        return np.dot(wstack.T, poly_coeffs)
