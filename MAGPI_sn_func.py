import numpy as np

def MAGPI_sn_func(index, signal=None, noise=None):
  """
  Return the S/N of of a bin with spaxles given by "index".
  
  This is a simple adaptation of the default sn_func used by 
  VorBin (https://pypi.org/project/vorbin) to include an additional
  scaling of the binned S/N to account for spatial covariance in the 
  reconstructed MAGPI data (see Mendel et al. in prep).
  """
  
  sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))
  sn /= 1. + 0.18*np.log10(index.size)
  
  return sn
