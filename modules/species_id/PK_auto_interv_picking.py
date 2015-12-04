'''
Created on 1 Jul 2015

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

import scipy.io.wavfile as spwave
import numpy as np
import pandas.stats.moments as mom
import matplotlib.pyplot as plt


import modules.misc.PK_stft as stft
import modules.species_id.PK_train_interv_picking as intvpck
import modules.species_id.PK_picking_helper as pckhlp


def calc_pattern_correlation_chunked(data,pattern,fs,freq_fft_bins ,chunk_len_s = 45,
                                        len_fft = 1024, nr_ffts_per_s = 100, pattern_len_s = 2):
    """
    Calculate the average correlation between the stft of a timeseries and a pattern over a certain 
    frequency range. Used for data generation for machine learning as input for peak finding algorithms.
    
    Parameters
    ----------
    data : 1D ndarray
        Timeseries
    pattern : ndarray
        The timeseries pattern which is used to calculate the correlation with data
    fs : float
        The sample frequency (frames per second) of the data      
    freq_fft_bins : list
        The frequency bins used for the correlation between pattern and the stft of the data
    chunk_len_s  : int
        The length in seconds for each chunked stft. 
    len_fft : int
        The length of each fft calculation. 
    nr_ffts_per_s : int
        Number of ffts per second in the stft. 
    pattern_len_s : int
        Length of the pattern in seconds
    Returns
    -------
    peaks : 1D ndarray
        The concatenated array of the correlation between data and pattern
 
    """
        
    
    n_frames_chunk,_,_,sec_per_sample,overlap = stft.calc_nr_frames(chunk_len_s,fs,len_fft,chunk_len_s*nr_ffts_per_s)
    # -- By giving the overlap, the length of the pattern is not necessarily pattern_len_s*nr_ffts_per_s anymore
    pattern = stft.calc_stft(pattern,0,pattern.shape[0], fs, pattern_len_s*nr_ffts_per_s,overlap=overlap)[0]
    # -- z-score the pattern
    pattern = (pattern-np.mean(pattern)) / np.std(pattern)
    #q75, q50, q25 = np.percentile(pattern, [75 ,50, 25])
    #iqr = q75 - q25
    #pattern  = 1/(1+np.exp(-(pattern -q50)/(iqr/1.35)))
    
#     plt.matshow(pattern, origin='lower')
#     exit()
    end_frame = 0
    start_frame = 0
    while end_frame < data.shape[0] - n_frames_chunk:
        start_frame = end_frame
        end_frame = end_frame+n_frames_chunk
        spectrum = stft.calc_stft(data,start_frame,end_frame, fs,chunk_len_s*nr_ffts_per_s)[0]
        #spectrum = (spectrum - np.mean(spectrum))/np.std(spectrum)
        print 'spectrum.shape: ',spectrum.shape
        
        for i in freq_fft_bins:
            if i == freq_fft_bins[0]:
                tmp = np.correlate(spectrum[i,:], pattern[i,:], mode='same', old_behavior=False)
                print 'tmp.shape: ',tmp.shape
                print 'spectrum[i,:].shape:' ,spectrum[i,:].shape
            else :
                tmp += np.correlate(spectrum[i,:], pattern[i,:], mode='same', old_behavior=False)
                print 'tmp.shape: ',tmp.shape  
                print 'spectrum[i,:].shape:' ,spectrum[i,:].shape
        if start_frame == 0:
            peaks = tmp
        else:
            peaks = np.hstack((peaks,tmp))
    
    return peaks





def compute_range_dict(source_file_path, pattern, config_par ,fs):
    """
    Compute the range dictionary for a single audio file for a given pattern. 
    Paramters:
    ----------
    source_file_path : string
        Path to the input wav file with sample frequency fs.
    pattern : ndarray
        The timeseries pattern which is used to calculate the correlation with data. In sample frequency fs.
    config_par : dict
        Dictionary containing all the required configuration parameter
    fs : float
        The sample frequency (frames per second) of the data  
    Returns:
    --------
    range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and end time
        for the training interval ranges.
    roll_max_peaks : ndarray
        Rolling maximum of data normalised by its rolling mean.
    """
    # -- Calculate the average correlation over all freq_range_fft_bins between source and pattern
    peaks = correlation_picking(source_file_path,pattern,config_par['freq_range_fft_bins'],
                                    len_fft = config_par["len_fft"],
                                    nr_ffts_per_s = config_par["nr_ffts_per_s"], 
                                    pattern_len_s = config_par["pattern_len_sec"]
                                    )
    
    # -- Identify frames of isolated peaks
    centre_frames,roll_max_peaks = peak_identification(peaks,
                                            width_in_s = config_par["peak_mute_width"],
                                            width_roll_mean = config_par["width_roll_mean"],
                                            roll_max_peaks_threshold = config_par['peaks_threshold'],
                                            fs = fs,nr_ffts_per_s = config_par["nr_ffts_per_s"],
                                            chunk_len_s = 60,len_fft = config_par['len_fft'],
                                            is_ret_roll_max_peaks = True)
    print 'centre_frames',centre_frames

    # -- Create range dictionary around isolated peaks
    range_dict = intvpck.training_range_times_from_peaktimes(centre_frames,
                                                config_par["range_len_sec"],source_file_path,fs)   
    return range_dict,roll_max_peaks
       
    
def correlation_picking(source_file_path,pattern,freq_range_fft_bins,len_fft = 1024, nr_ffts_per_s = 100, pattern_len_s = 2):
    """
    Calculate the correlation between the timeseries given in a wav file
    with the pattern given in a 1D array.
    Parameter:
    ----------
    source_file_path : string
        Path to the wav file acting as source
    pattern : ndarray
        The pattern which is used to calculate the correlation with data
    freq_range_fft_bins : list
        The frequency bins used for the correlation between pattern and the stft of the data    
    len_fft : int
        The length of each fft calculation. 
    nr_ffts_per_s : int
        Number of ffts per second in the stft. 
    pattern_len_s : int
        Length of the pattern in seconds
    Returns:
    --------
    peaks : 1D ndarray
        The array of the correlation between source and pattern. The samplerate is not identical with the sample
        rate of the input file but depends on nr_ffts_per_s given to calc_pattern_correlation_chunked
    """
    # -- read audio data
    fs ,data = spwave.read(source_file_path)
    # -- z-score data
    data = (data-np.mean(data))/np.std(data)
    peaks = calc_pattern_correlation_chunked(data,pattern,fs,
                            freq_range_fft_bins,len_fft=len_fft, 
                            nr_ffts_per_s = nr_ffts_per_s, pattern_len_s = pattern_len_s)
    
    
    
    return peaks

   
def find_peak_ind(data,width,width_roll_mean = 200,roll_max_peaks_threshold = 4.0, is_ret_roll_max_peaks = False):
    """
    Calculate the indices of isolated maxima in the data array usually containing the result
    of a correlation calculation bewteen a timeseries and a pattern.
    
    Parameters
    ----------
    data : 1d ndarray
        Timeseries,usually containing the result
        of a correlation calculation between a timeseries and a pattern.
    width : int
        The width of an interval in which the maximum is found. I.e. two maxima have to be at least
        width apart to be registered as separate.
    width_roll_mean : int
        The width used for the rolling mean normalisation of the data for better identification
        of pattern matches as it only looks for narrow peaks.
    roll_max_peaks_threshold : float
        The threshold for when a peak is considered high enough to be added to the returned indices.
        A peak has to be roll_max_peaks_threshold times larger in amplitude than the rolling mean to be
        registered as valid peak.
    is_ret_roll_max_peaks : bool
        Return roll_max_peaks or not. Default is not.
    Returns
    -------
    peak_inds : list
        List of indices of the peaks in data.
    roll_max_peaks : ndarray, if is_ret_roll_max_peaks
        Rolling maximum of data normalised by its rolling mean.
    """

    roll_mean = mom.rolling_mean(data, width_roll_mean,center=True)
#     plt.figure()
#     plt.plot(data)
#     plt.show()
    roll_mean = 1
    roll_max_peaks = mom.rolling_max(data/roll_mean,width,center=False)
    # -- Calculate the centered rolling max. 
    roll_max_peaks_c = mom.rolling_max(data/roll_mean,width,center=True)    

    roll_peak_inds, = np.nonzero((roll_max_peaks > roll_max_peaks_threshold))
    peak_inds = []
    for c in roll_peak_inds[1:-1]:
        # -- max is when left entry in roll_max_peaks is smaller and right is equal and
        #    if in centered roll_max_peaks_c the left (and the right) are the same
        if (roll_max_peaks[c-1] < roll_max_peaks[c] and np.abs(roll_max_peaks[c]-roll_max_peaks[c+1]) < 0.0001
                and np.abs(roll_max_peaks[c]-roll_max_peaks_c[c-1]) < 0.0001):
            peak_inds.append(c)
            
    if is_ret_roll_max_peaks:
        return peak_inds,roll_max_peaks
    else:
        return peak_inds


def peak_identification(peaks,width_in_s,width_roll_mean = 200,
                        roll_max_peaks_threshold = 4.0,fs = 16000,nr_ffts_per_s = 100,
                        chunk_len_s = 60,len_fft = 1024,is_ret_roll_max_peaks = False):
    """
    Identify isolated peaks in an 1d array calculated by correlation_picking.
    Parameters:
    -----------
    peaks : ndarary
        Array containing isolated peaks with a sample rate depending on fft_per_sec = 100
    width_in_s : int
        The width in seconds of an interval in which the maximum is found. I.e. two maxima have to be at least
        width_in_s apart to be registered as separate.
    width_roll_mean : int
        The width used for the rolling mean normalisation of the data for better identification
        of pattern matches as it only looks for narrow peaks.
    roll_max_peaks_threshold : float
        The threshold for when a peak is considered high enough to be added to the returned indices.
        A peak has to be roll_max_peaks_threshold times larger in amplitude than the rolling mean to be
        registered as valid peak.    
    fs : float
        The sample frequency (frames per second) of the data       
    nr_ffts_per_s : int
        Number of ffts per second in the stft.
    chunk_len_s  : int
        The length in seconds for each chunked stft. 
    len_fft : int
        The length of each fft calculation.     
    is_ret_roll_max_peaks : bool
        Return roll_max_peaks or not. Default is not.
    Returns:
    --------
    peak_frame_list : list
        List of frames in the original sound file used in correlation_picking() containing peaks   
    roll_max_peaks : ndarray, if is_ret_roll_max_peaks
        Rolling maximum of data normalised by its rolling mean.
    """


    _,_,frames_per_sample,sec_per_sample,_ = stft.calc_nr_frames(chunk_len_s,fs,len_fft,chunk_len_s*nr_ffts_per_s)
    if is_ret_roll_max_peaks:
        inds,roll_max_peaks = find_peak_ind(peaks,width_in_s/sec_per_sample,width_roll_mean = width_roll_mean,
                                            roll_max_peaks_threshold = roll_max_peaks_threshold,is_ret_roll_max_peaks = True)
    else:
        inds = find_peak_ind(peaks,width_in_s/sec_per_sample,width_roll_mean = width_roll_mean,roll_max_peaks_threshold = roll_max_peaks_threshold)
    peak_frame_list = np.array([ind*frames_per_sample for ind in inds])
    
    if is_ret_roll_max_peaks:
        return peak_frame_list,roll_max_peaks
    else:
        return peak_frame_list

def plot_from_range_dict(range_dict,pdf_path,nr_fft = 100, 
                                len_fft = 1024, nr_mel_bins = 100):
    """
    Plot the ranges in the range_dict in mel spectrum scale and save to file
    Parameters:
    -----------
    range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and end time
        for the training interval ranges.
    pdf_path : string
        Path to a location where a pdf file is created containing the plots for all
        ranges in range_dict  
    nr_fft : int
        The number of ffts in each stft calculation
    len_fft : int
        The length of each of the nr_fft fourier transforms for each stft 
    nr_mel_bins : int
        The number of bins in which the mel spectrum is to be divided in
    Results:
    --------
    None
    """
    X = intvpck.train_interval_mel_features(range_dict, nr_fft = nr_fft, 
                                len_fft = len_fft, nr_mel_bins = nr_mel_bins)
    plot_titles_lst = pckhlp.plot_titles_list_from_dict(range_dict)
    pckhlp.save_feat_in_pdf(X, (nr_mel_bins,nr_fft), pdf_path,plot_titles_lst)
    
    return None
