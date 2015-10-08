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
import itertools

import glob
import re
import os.path as op

from scipy import stats
import modules.misc.PK_matlab_IO as mIO

def filter_calculated(root_dir):
    """
    Return all problems for which the u statistic values havent been previously
    calculated
    Parameters:
    -----------
    root_dir : string
        Directory containing files to be investigated
            
    Returns
    -------
    retval : list of strings
        Names of problems for which u statistic values have yet to be calculated
    """
    calced_names,dat_names = get_calculated_names(root_dir)
    return list(set(dat_names) - set(calced_names))


def get_calculated_names(root_dir):
    """
    Return the names of of the problems with previously calculated u statistic 
    values and the names of the data matices (HCTSA_loc.mat format) in root_dir
    Parameters:
    -----------
    root_dir : string
        Directory containing files to be investigated
    Returns:
    --------
    calced_names,dat_names : list of strings
        Names of problems with previously calculated u statistic values and names
        of all problems (HCTSA_loc.mat format) in the root_dir
    """
    file_paths =  glob.glob(root_dir+'/*')
    file_names = [op.basename(file_path) for file_path in file_paths]  
    calced_names = []
    dat_names = []
    for file_name in file_names:
        if re.search('.*_ustat\.npy',file_name) != None:
            calced_names.append(re.search('(.*)_ustat\.npy',file_name).group(1))
        if re.search('HCTSA_.*_N_70_100_reduced.mat',file_name) != None:
            dat_names.append(re.search('HCTSA_(.*)_N_70_100_reduced.mat',file_name).group(1)) 
    return  calced_names,dat_names
        

def u_stat_norm_factor(file_name,is_from_old_matlab = False):
    """
    Return the u statisitc scaling factor n_1*n_2 where n_i is the 
    number of time series with label i. Every entry corresponds to 
    one label pairing in the classification problem pointed to 
    by file_name.
    Parameters:
    -----------
    file_name : string
        Filename of HCTSA_loc.mat file containing data of the current problem    
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.

    Returns:
    --------
    u_scale : array
        Scaling factor for u statistic for every label pairing in the current
        classification problem
    """
    ts, = mIO.read_from_mat_file(file_name,['TimeSeries'],is_from_old_matlab = is_from_old_matlab )
    labels = [int(x.split(',')[-1]) for x in ts['keywords']]
    
    labels_unique = list(set(labels))
    labels = np.array(labels)
    n_labels = len(labels_unique)
    nr_items_label = []
    
    # -- for every label calculate the number of time series with this label
    for i,label in enumerate(labels_unique):
        nr_items_label.append(np.nonzero((labels == label))[0].shape[0])
    
    # -- initialise the u_scale array for all label pairings    
    u_scale = np.zeros(n_labels * (n_labels-1)/2 )
    
    # -- calculate the scaling factor for the u statistic for every label pairing
    for i,(label_ind_0,label_ind_1) in enumerate(itertools.combinations(range(n_labels),2)):
        u_scale[i] = nr_items_label[label_ind_0]*nr_items_label[label_ind_1]
        
    return u_scale



def u_stat_all_label(ts,data,mask=None):
    """
    Calculate the U-statistic for all label pairings in a HCTSA_loc.mat file. The operations can be masked 
    by a boolean array.
    Parameters:
    -----------
    ts : dict
        'TimeSeries' dictionary from HCTSA_loc.mat file.
    data : ndarray
        data array where each row represents a timeseries and each column represents a feature
    mask : ndarray dtype='bool', optional
        If given this acts as mask to which features are included in the calculation
    Returns:
    --------
    ranks : ndarray of dim (n_labels * (n_labels-1) / 2, nr_features)
        The U-statistic for all features and label pairs, where a row represents a pair of labels.
        Every column represents one feature.
    labels_unique : list
        List of all unique labels
    label_ind_list : list 
        List of lists where each sub-list i represents all row-indices in data containing timeseries
        for labels_unique[i]
    """
    # ---------------------------------------------------------------------
    # mask the data if required
    # ---------------------------------------------------------------------   
    if mask != None:
        data = data[:,mask]
    # ---------------------------------------------------------------------
    # extract the unique labels
    # ---------------------------------------------------------------------
    print ts['keywords']
    labels = [x.split(',')[-1][0] for x in ts['keywords']]
    labels_unique = list(set(labels))
    
    labels = np.array(labels,dtype = np.dtype('S2'))
    label_ind_list = []
    
    # ---------------------------------------------------------------------
    # get indices for all unique labels
    # ---------------------------------------------------------------------
    for i,label in enumerate(labels_unique):
        label_ind_list.append(np.nonzero((labels == label))[0])
    n_labels = len(label_ind_list)
    
    # ---------------------------------------------------------------------
    # calculate Mann-Whitney u-test
    # ---------------------------------------------------------------------
    ranks = np.zeros((n_labels * (n_labels-1) / 2,data.shape[1]))
    
    for i,(label_ind_0,label_ind_1) in enumerate(itertools.combinations(range(n_labels),2)):
        data_0 = data[label_ind_list[label_ind_0],:]
        #print label_ind_list[label_ind_0]
        #print data_0
        data_1 = data[label_ind_list[label_ind_1],:]
        print i+1,'/',n_labels * (n_labels-1) / 2
        for k in range(0,data.shape[1]):
            if np.array_equiv(np.concatenate((data_0[:,k],data_1[:,k])),data_0[0,k]):
                ranks[i,k] = data_0[:,k].shape[0] * data_1[:,k].shape[0]/2.
            else:
                ranks[i,k] = stats.mannwhitneyu(data_0[:,k], data_1[:,k])[0]

    return ranks,labels_unique,label_ind_list
 
def u_stat_all_label_file_name(file_name,mask = None, is_from_old_matlab=False):
    """
    Calculate the U-statistic for all label pairings in a HCTSA_loc.mat file. The operations can be masked 
    by a boolean array.
    Parameters:
    -----------
    file_name : string
        File name of the HCTSA_loc.mat file containing as least the matrices 'TimeSeries' and 'TS_DataMat'
    mask : ndarray dtype='bool', optional
        If given this acts as mask to which features are included in the calculation
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.

    Returns:
    --------
    ranks : ndarray of dim (n_labels * (n_labels-1) / 2, nr_timeseries)
        The U-statistic for all features and label pairs, where a row represents a pair of labels.
        Every column represents one feature    
    labels_unique : list
        List of all unique labels
    label_ind_list : list 
        List of lists where each sub-list i represents all row-indices in data containing timeseries
        for labels_unique[i]
    """
    # ---------------------------------------------------------------------
    # load the data
    # ---------------------------------------------------------------------
    ts,data = mIO.read_from_mat_file(file_name,['TimeSeries','TS_DataMat'],is_from_old_matlab = is_from_old_matlab )
    
    # ---------------------------------------------------------------------
    # mask the data if required
    # ---------------------------------------------------------------------   
    if mask != None:
        data = data[:,mask]
    
    # ---------------------------------------------------------------------
    # extract the unique labels
    # ---------------------------------------------------------------------
    labels = [int(x.split(',')[-1]) for x in ts['keywords']]
    labels_unique = list(set(labels))
    
    labels = np.array(labels)
    label_ind_list = []
    
    # ---------------------------------------------------------------------
    # get indices for all unique labels
    # ---------------------------------------------------------------------
    for i,label in enumerate(labels_unique):
        label_ind_list.append(np.nonzero((labels == label))[0])
    n_labels = len(label_ind_list)
    
    # ---------------------------------------------------------------------
    # calculate Mann-Whitney u-test
    # ---------------------------------------------------------------------
    ranks = np.zeros((n_labels * (n_labels-1) / 2,data.shape[1]))
    
    for i,(label_ind_0,label_ind_1) in enumerate(itertools.combinations(range(n_labels),2)):
        # -- select the data for the current labels
        data_0 = data[label_ind_list[label_ind_0],:]
        data_1 = data[label_ind_list[label_ind_1],:]
        print i,'/',n_labels * (n_labels-1) / 2
        for k in range(0,data.shape[1]):
            # -- in the case of same value for every feature in both arrays set max possible value
            if np.array_equiv(np.concatenate((data_0[:,k],data_1[:,k])),data_0[0,k]):
                ranks[i,k] = data_0[:,k].shape[0] * data_1[:,k].shape[0]/2.
            else:
                ranks[i,k] = stats.mannwhitneyu(data_0[:,k], data_1[:,k])[0]

    return ranks,labels_unique,label_ind_list