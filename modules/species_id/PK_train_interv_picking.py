'''
Created on 17 Jun 2015

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
import scipy.io.wavfile as spwav
import modules.misc.PK_stft as stft
import modules.species_id.PK_mel as mel
import matplotlib.pyplot as plt

def training_range_times_from_peaktimes(peaks,range_witdth_s,file_path,fs):
    """
    Create a training interval range dict from a list of peak frames in a specified audio file
    Parameters:
    -----------
    peaks : array like
        Frames of the peaks in the given file
    range_width_s : float
        width of the training range in seconds
    file_path : string
        The path to the audiofile used for the peaks array
    fs : float
        Sampling frequency of the audio file pointed to by file_path
    Returns:
    --------    
    interval_range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and end time
        for the training interval ranges.
    """
    # -- Create the interval_range_dict to be filled
    interval_range_dict = {file_path : []}
    for peak_frame in peaks:
        # -- time interval from peak frame rounded to 10ms
        st = round( peak_frame/fs - range_witdth_s/2. , 2)
        et = st + range_witdth_s
        interval_range_dict[file_path].append([st,et])
    return interval_range_dict

def training_range_times_from_file(timefile_path, nr_header_lines = 2, min_length_s = 1.,max_overshoot_s = 0.):
    """
    Create a training interval range dict from a text file containing audiofile name 
    and start and end time.
    
    Parameters:
    -----------
    timefile_path : string
        Path to the textfile containing audiofile name, start and end time.
    nr_header_lines : int
        Number of lines to be ignored at the beginning of the file timefile
    min_length_s : float
        The minimum time in seconds an interval range has to be to be included (usually the length of 
        the training intervals)
    max_overshoot_s : float
        The maximum time outside the given interval range allowed to be included in the final range.
        E.g. 0.5 means the training intervals can start 0.5 seconds before the given interval range start time
        and end up to 0.5s after the end of the given interval ranges.
    Returns:
    --------
    interval_range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and endtime
        for the training interval ranges.
    """
  
    # -- load the the timing file
    with open(timefile_path,'r') as timefile:
        content_timefile = timefile.readlines()
        # -- convert the read lines into a list containing the required data. Skip 2 header lines. remove empty lines
        list_timefile = [[t(s) for t,s in zip((str,float,float,int),line.split())] for line in content_timefile[nr_header_lines:] if line.strip()]
        # -- check that all intervals are at least min_length_s seconds long
        list_timefile = [item for item in list_timefile if (item[2]-item[1]) >= min_length_s]
    
    interval_range_dict = {}
    
    for [filename,starttime,endtime,_]  in list_timefile:
        starttime = np.min(starttime - max_overshoot_s,0.)
        # -- this might be after the end of the file as there is no way of knowing how long the recoring is
        #    at this point
        endtime += max_overshoot_s

        if not filename in interval_range_dict:
            interval_range_dict[filename] = [[starttime,endtime]]
        else:
            interval_range_dict[filename].append([starttime,endtime])
        
    return interval_range_dict


def training_interval_times(interval_range_dict,nr_intervals_per_label,interval_length_s):
    """
    Compute a dictionary containing training intervals of interval_length_s randomly selected in
    the given interval ranges. If possible one sample is chosen in each interval range.
    Parameters:
    -----------
    intervals_range_dict : dict
        Dictionary containing the interval ranges as given by training_interval_range_from_file.
    nr_intervals_per_label : int
        Number of training intervals returned 
    interval_length_s : int
        Length of the returned training intervals in seconds
    Returns:
    --------
    interval_dict : dict
        Dictionary containing the file_names as key and a list of lists of [start_time,end_time] for each interval
    """
    # -- Prepare a list of 3-tuples (name,start,end) for random picking
    interval_range_list = []
    for key in interval_range_dict:
        interval_ranges_curr = interval_range_dict[key]
        interval_ranges_file_curr =  zip([key]*len(interval_ranges_curr),interval_ranges_curr)
        interval_range_list += interval_ranges_file_curr
    n_interval_ranges = len(interval_range_list)    
    # -- for the smaller of either the number of samples per label or the number of interval_ranges of training sets
    # -- pick without replacement until enough intervals have been picked, then pick with replacement randomly
    n_unique = min(nr_intervals_per_label,n_interval_ranges)
    rand_ind_unique = np.random.choice(n_interval_ranges,size=n_unique, replace=False)
    # if we want more training intervals than there are interval_ranges
    if nr_intervals_per_label > n_interval_ranges:
        rand_inds = np.concatenate((rand_ind_unique,np.random.choice(n_interval_ranges,size=nr_intervals_per_label-n_unique, replace=True)))
    else:
        rand_inds = rand_ind_unique
    interval_dict = {key:[] for key in interval_range_dict}
    
    for rand_ind in rand_inds:
        # -- compute the maximum starting time for this interval
        max_start_time = interval_range_list[rand_ind][1][1] - interval_length_s
        start_time = np.random.uniform(interval_range_list[rand_ind][1][0],max_start_time)
        end_time = start_time + interval_length_s
        
        interval_dict[interval_range_list[rand_ind][0]] += [[round(start_time,2),round(end_time,2)]]

    return interval_dict

def train_interval_mel_features(interval_dict,pattern = '{:s}',nr_fft = 100, 
                                len_fft = 1024, nr_mel_bins = 100, min_freq_wanted = 200, max_freq_wanted = 8000,
                                is_return_fit = False):
    """
    Calculate the mel spectrogram features for each interval in the interval_dict and return a feature matrix X
    of ndim = [nr_intervals,nr_features] 
    Parameters:
    -----------
    interval_dict : dict
        Dictionary which keys point to a file_path via pattern and which values are list of lists of [start_time,end_time]
        for trinaing intervals
    pattern : string
        A formatting string to map from the interval_dict keys to file_paths
    nr_fft : int
        The number of ffts in each stft calculation
    len_fft : int
        The length of each of the nr_fft fourier transforms for each stft 
    nr_mel_bins : int
        The number of bins in which the mel spectrum is to be divided in
    min_freq_wanted, max_freq_wanted : float
        The lowest/highest frequency in the returned mel spectrum
        
    Returns:
    --------
    X : ndarray
        Array containing the flattened stft in mel spectrum for each interval in interval_dict.
        Each row corresponds to one interval and each colum to one feature of the flattened mel spectrum
    """
    # -- The total number of intervals in the dictionary 
    nr_intervals = np.array([len(interval_dict[file_key]) for file_key in interval_dict]).sum()
    X = np.zeros((nr_intervals,nr_mel_bins*nr_fft))
    t = np.zeros((nr_intervals,2))
    i=0
    file_interval_tuple = ()
    for file_key in interval_dict:
        file_name = pattern.format(file_key)
        print file_name
        fs,data = spwav.read(file_name)
        data=data-np.mean(data)
        for time_interval in interval_dict[file_key]:

            file_interval_tuple = file_interval_tuple+((file_key,time_interval),)
            sf = int(time_interval[0]*fs)
            ef = int(time_interval[1]*fs)
            spectrum = stft.calc_stft(data[sf:ef],fs = fs, nr_fft = nr_fft, len_fft = len_fft)[0]
            X[i,:] = mel.stft_to_mel_freq(spectrum,fs=fs,len_fft=len_fft,nr_mel_bins = nr_mel_bins, 
                                             min_freq_wanted = min_freq_wanted , max_freq_wanted= max_freq_wanted)[0].flatten()  
            t[i] = time_interval
            i+=1 
    if is_return_fit:
        return X,file_interval_tuple
    
    return X

def write_range_dict_to_txt(range_dict, out_file_path):
    """
    Write a range dictionary or an interval dictionary to a file in the format used 
    by training_range_times_from_file()
    Parameters:
    -----------
    range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and end time
        for the training interval ranges.
    out_file_path : string
        Path to the output text file.   
    Returns:
    --------
    None
    
    """
    with open(out_file_path,'w') as f:
        f.write("FileKey\t\tStartTime\t\tEndTime\t\tType\n")
        f.write("---------------------------------------------------------------------\n")
        for key in range_dict:
            for item in range_dict[key]:
                f.write("{:s}\t{:.2f}\t{:.2f}\t0\n".format(key,item[0],item[1]))
    return None
                
                
                
                
                
if __name__ == "__main__":
    #timefile_path = '/home/philip/work/bio_diversity/sample_recordings_species/fluffy-backed_tit-babbler/xeno_canto/wav/Times_FBTB.txt'
    #range_dict = training_range_times_from_file(timefile_path, nr_header_lines = 2, min_length_s = 1.)
    
    centre_frames = np.load('/home/philip/workspace/speciesID/src/data_picking/centre_frames.npy')
    file_path = '/home/philip/work/bio_diversity/recordings/data/SM3SET3__0__20150424_062134.wav'
    range_dict = training_range_times_from_peaktimes(centre_frames,1.,file_path,16e3)
    interval_dict = training_interval_times(range_dict,50,1)
    print ['[{:d}:{:f},{:d}:{:f}]'.format(int(st/60),np.mod(st,60),int(et/60),np.mod(et,60)) for [st,et] in interval_dict[file_path]]

    pattern = '{:s}'
    #pattern = '/home/philip/work/bio_diversity/sample_recordings_species/fluffy-backed_tit-babbler/xeno_canto/wav/{:s}.wav'
    X = train_interval_mel_features(interval_dict, pattern = pattern)
    np.save('/home/philip/Desktop/tmp/test_data_pec_id/X.npy',X)
    #plt.plot(X[-1,:])
    plt.matshow(X[-1,:].reshape((100,100))[20:40,:],origin='lower')
    plt.show()
    
    #interval_range_list = []
    
    #for key in dict:
    #    interval_range_list += dict[key]

#     # -- for a certain number of training sets
#     # -- pick without replacement until all have been picked, then pick with replacement randomly
#     n_unique = min(n_samples_per_label,n_intervals)
#     rand_ind_unique = np.random.choice(len(list_timefile),size=n_unique, replace=False)
#     # if we want more training sets than there are intervals
#     if n_samples_per_label > n_intervals:
#         rand_ind = np.concatenate((rand_ind_unique,np.random.choice(len(list_timefile),size=n_samples_per_label-n_unique, replace=True)))
#     else:
#         rand_ind = rand_ind_unique