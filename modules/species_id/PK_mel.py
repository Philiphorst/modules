'''
Created on 10 Jun 2015

@author: philip knaute
------------------------------------------------------------------------------
Copyright (C) 2015, Philip Knaute <philiphorst.project@gmail.com>,

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
California, 94041, USA.
------------------------------------------------------------------------------
'''
import numpy as np
import scipy.interpolate as interp

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.0)
  
def interp_1(x, y, z) :
    """
    Interpolate and subsample an array along axis = 0 
    (columns are vectors to be interpolated)
    Parameters
    ----------
    x : ndarray
        Original data
    y : 1d ndarray
        Locations of samples of x
    z : 1d ndarray
        Locations for the interpolated data which is returned
    Returns
    -------
    out : ndarray
        Result of interpolating x on the locations given in z
    """
    rows, cols = x.shape
    out = np.empty(z.shape + (cols,))
    for j in xrange(cols) :
        out[:,j] = interp.interp1d(y, x[:,j],bounds_error=False)(z)
    return out  
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def stft_to_mel_freq(data,**kwargs):
    """ 
    transform stft data to mel frequency scale
    Parameters
    ----------
    data : ndarray
        stft array
    fs : float
        sampling frequency of the timeseries
    len_fft : int
        length of the fourier transform
    n_mel_bins : int
        number of the bins in the for the mel transform
    min_freq_wanted, max_freq_wanted : float (optional)
        min and max frequency included in the mel transform
        Can be considered similar to boxcar bandpass filter
    Returns
    -------
    retval : ndarray
        Stft sampled at mel-frequency bins
        
    """
    # ----------------------------
    # -- extract keyword arguments
    # ----------------------------
    fs = kwargs.get('fs')
    len_fft =  kwargs.get('len_fft')
    nr_mel_bins =  kwargs.get('nr_mel_bins')
    min_freq_wanted =  kwargs.get('min_freq_wanted', None) 
    max_freq_wanted =  kwargs.get('max_freq_wanted', None) 
    # ----------------------------
    
    bin_hz = np.linspace(0,fs/2+1,len_fft/2+1)
    if min_freq_wanted == None:
        min_freq_wanted = 0
    if  (max_freq_wanted == None) or (max_freq_wanted > fs/2+1):
        max_freq_wanted = fs/2+1
        
    bin_mel = np.linspace(min_freq_wanted,hz2mel(max_freq_wanted),nr_mel_bins)
    bin_mel_hz = mel2hz(bin_mel)
    # -- make sure the max frequency in bin_mel_hz is equal or smaller than
    #    the nyquist frequency to avoid NaN entries in the interpolated array
    bin_mel_hz[-1] = min(bin_mel_hz[-1],bin_hz[-1])
    return interp_1(data, bin_hz, bin_mel_hz),bin_mel_hz
    
def compute_mel_freq(fs,len_fft = 1024,nr_mel_bins = 100,min_freq_wanted = None , max_freq_wanted= None):
    """ 
    Compute mel frequency scale in frequency band
    """

    bin_hz = np.linspace(0,fs/2+1,len_fft/2+1)
    if min_freq_wanted == None:
        min_freq_wanted = 0
    if  (max_freq_wanted == None) or (max_freq_wanted > fs/2+1):
        max_freq_wanted = fs/2+1
        
    bin_mel = np.linspace(min_freq_wanted,hz2mel(max_freq_wanted),nr_mel_bins)
    bin_mel_hz = mel2hz(bin_mel)
    # -- make sure the max frequency in bin_mel_hz is equal or smaller than
    #    the nyquist frequency to avoid NaN entries in the interpolated array
    bin_mel_hz[-1] = min(bin_mel_hz[-1],bin_hz[-1])
    return bin_mel_hz 