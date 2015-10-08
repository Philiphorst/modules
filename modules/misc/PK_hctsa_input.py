'''
Created on 26 Aug 2015

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
import re
import scipy.io.wavfile as spwave
import numpy as np
import os

   
def write_data_from_file_paths(file_paths,ts_in_path,out_path,re_kw_pattern, is_write_INP_ts_only=False,int_len_sec=None):
    """
    Create plain text data files and hctsa TS_INP.txt file for all files pointed to by file_paths
    
    Paramters:
    ----------
    file_paths: list
        file paths of wav files to be converted
    ts_in_path: string
        path to hctsa TS_INP.txt file which will be written to
    out_path: string
        path to output folder for writing converted files to
    re_kw_pattern: string
        regular expression pattern to extract keywords from the file_paths of the input sound files
    out_fs: int
        sampling rate the data is ressmpled to
    is_write_INP_ts_only : bool
        If true writes only INP_ts.txt files for existing data.txt files. This has to be matching. Used mainly if number of
        processors used changes.
    int_len_sec: double
        max length in seconds of output data files. Random segments are chosen if given.

    Returns:
    --------
    
    """
    with open(ts_in_path,'w') as ts_in:
        for file_path in file_paths:
            
            # -- create the keywords list  
            print file_path
            m = re.match(re_kw_pattern,file_path)
            keywords_list = [item for item in m.groups()]
            
            # -- create path to the output file
            file_name = os.path.basename(file_path)   
            out_file_path = out_path+file_name.split('.')[0]+'.txt'
            
            if not is_write_INP_ts_only:
                # -- read the sound file
                fs,data = spwave.read(file_path)
            
                # -- pick random interval of length int_len_sec if not None
                if int_len_sec != None:
                    start_frame = np.random.randint(0,data.shape[0]-int_len_sec*fs)
                    data = data[start_frame:start_frame+int_len_sec*fs]

                # -- save the data as text file    
                np.savetxt(out_file_path, data)
            
            # -- add line to matlab timeseries input file
            ts_in.write('{:s}\t\t{:s},{:s},{:s}\n'.format(out_file_path,*keywords_list))
            
            
def write_op_name_txt(outfile_path,op_names_list):
    """
    Write a plain text file containing a set of operation names
    Parameters:
    -----------
    outfile_path : string
        Path to the plain textfile to which the operation names are written
    op_names_list : iterable
        Iterable over strings containing the operation names
    Returns:
    --------
        None
    """
    with open(outfile_path,'w') as outfile:
        for op_name in op_names_list:
            outfile.write(op_name+'\n')