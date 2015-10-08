'''
Created on 8 Jun 2015

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
import matplotlib.mlab as mlab
import numpy as np

def calc_stft(data,start_frame=0,end_frame=None, fs = 16000, nr_fft = 100, overlap = None, len_fft = 1024, is_dB = False):
    """
    Calculate the short time fourier transform using mlab.specgram for an interval 
    in the time-series 1D-array given as data.
    Parameters
    ----------
    data : 1-dim ndarray
        Timeseries for which the stft is to be calculated
    start_frame,end_frame : int
        The start and end indices in data that limit the interval of the stft
    fs : float
        The sample frequency (frames per second) of the data
    nr_fft : int
        The number of ffts to be calculated over the interval.
    overlap : int (optional)
        The number of frames that consecutive intervals overlap. If given will 
        overwrite nr_fft. If not given will be calculated using nr_fft. 
        Default is to calculate using nr_fft.
    len_fft : int
        The length of each fft calculation.
    is_dB : bool
        Is the spectrum retorned on a dB scale.
        
    Returns
    -------
    spectrum : ndarray
        The time dependent psd of the signal
    overlap : int
        The overlap used (see Parameters)
    """
    
    x = data[start_frame:end_frame]
    if overlap == None:
        overlap = int(np.ceil(len_fft-(x.shape[0] - len_fft)/(nr_fft -1)))
    spectrum = mlab.specgram(np.array(x), NFFT=1024, Fs=fs,noverlap=overlap)[0]
    if is_dB:
        spectrum[spectrum<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
        spectrum = 20 * np.log10(spectrum)  
    
    return spectrum,overlap

def calc_nr_frames(time_s,fs,len_fft,nr_fft):
    """
    Calculate the number of frames that can be covered perfectly by intervals used in an stft 
    using the parameters given.
    
    Parameters
    ----------
    time_s : float
        Interval length in seconds
    fs : float
        Sample frequency (frames per second)
    nr_fft : int
        The number of ffts to be calculated over the interval.
    len_fft : int
        The length of each fft calculation.
        
    Returns
    -------
    nr_frames : int
        number of frames covered by stft intervals created by using the parameters.
    in_s : float
        the length of the resulting interval in seconds.
    -------
        
    """
#    nr_frames = int(np.floor((time_s * fs - len_fft) / (nr_fft - 1))) *(nr_fft - 1) + len_fft
    overlap = int(np.ceil(len_fft - (time_s*fs - len_fft)/(nr_fft - 1)))
    nr_frames = len_fft + (len_fft-overlap)*(nr_fft - 1)
    in_s = nr_frames/float(fs)
    frames_per_sample = nr_frames/float(nr_fft)
    sec_per_sample = frames_per_sample/fs
    return nr_frames,in_s,frames_per_sample,sec_per_sample,overlap