'''
Created on 25 Jun 2015

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
import glob
import re
import os.path as osp

import modules.misc.PK_matlab_IO as mIO

def best_noncorr_op_ind(ind_dict,mask,file_path,op = None,is_from_old_matlab = False):
    """
    Compute the indices for the top features for a specific HCTSA_loc.mat file
    and the corresponding operation ids
    Parameters:
    -----------
    ind_dict : dict
        Dictionary where keys are file paths and values are the indices in the data matrix 
        of HCTSA_loc.mat
    mask : array like
        Mask to reduce the indices given in ind_dict
    file_path : string
        Path to HCTSA_loc.mat file
    op : dict,optional
        Operations dictionary from HCTSA_loc.mat file at file_path 
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.

    Returns:
    --------
    ind_top : array
        Indices of the features combining the information ind_dict and mask for the HCTSA_loc.mat
        file pointed to by file_path
    op_id_top : array
        Operation ids corresponding to ind_top
    """
    ind = np.array(ind_dict[file_path])
    if op == None:
        op, = mIO.read_from_mat_file(file_path, ['Operations'],is_from_old_matlab = is_from_old_matlab)
    op_id_top = np.array(op['id'])[ind][mask]
    ind_top = ind[mask]
    return ind_top,op_id_top

def count_op_calc(mat_file_paths,is_from_old_matlab = False):
    """
    Counts how many times every operation has been calculated successfully for each problem
    represented by a HCTSA_loc.mat file in root_dir
    Parameters:
    ----------
    mat_file_paths : list
        Paths to the HCTSA_loc.mat files corresponding to the problems considered.   
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.
    Returns:
    --------
    count_op_calc_all_problems : ndarray
        Array where each entry represents one operation and each value is the number of successful
        calculations of the corresponding operation for the given problems.
    """    

    count_op_calc_all_problems = np.zeros(10000)
    for mat_file_path in mat_file_paths:
        
        op, = mIO.read_from_mat_file(mat_file_path,['Operations'],is_from_old_matlab = is_from_old_matlab )
        print "Counting which operations calculated in: {:s}".format(mat_file_path)
        
        count_op_calc_all_problems[op['id']]+=1
    return count_op_calc_all_problems