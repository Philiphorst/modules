'''
Created on 30 Jun 2015

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

def all_files_in_dir_subdir(parent_path):
    """
    Find paths to all files in the directory under the parent path and all sub directories of depth one.
    Parameter:
    ----------
    parent_path: string
        The path to the parent directory
    Returns:
    --------
    file_paths: list
        list containing paths to all files in the parent and all subdirectories
    """
    parent_file_paths = glob.glob(parent_path+'/*')

    file_paths = []
    for path in parent_file_paths:
        file_paths +=  glob.glob(path+'/*')
    return file_paths

def ind_map_subset(a,b,subset_a):
    """
    Given two lists that are connected by their indices. I.E. if zip(a,b) would make sense, pick the 
    subset in b hat corresponds to the subset in a subset_a. 
    Parameters:
    -----------
    a,b : lists
        Lists connected by their indices
    subset_a : list
        Subset of entries in a
    Returns:
    -------
    subset_b : list
        The subset of b that corresponds to the subset subset_a of a.
    """
    # -- cast all arrays to lists
    if isinstance(a,np.ndarray):
        a = a.tolist()
    if isinstance(b,np.ndarray):
        b = b.tolist()
    if isinstance(subset_a,np.ndarray):
        subset_a = subset_a.tolist()       
    # --check if subset_a is a scalar
    if not hasattr(subset_a, "__len__"):
        subset_b = b[a.index(subset_a)]
    else:
        subset_b = [b[a.index(item)] for item in subset_a]
    # -- check if return value is scalar
    if not hasattr(subset_b, "__len__"):
        return subset_b
    else:
        return subset_b
    
def ismember(a, b,is_return_masked_array = False,return_dtype = int):
    """ Simulate Matlab's ismember function.
    Calculate lowest index in b for each value in a that is a member of b. fill if value in a is not
        member of b. 
        For masked arrays as input: masked values in a return fill as well as items in
        a that have only masked equivalents in b.
    Parameters:
    -----------
    a,b : list,ndarray,masked ndarray
    is_return_masked_array: bool
        Return a masked array instead of a list. Invalid values will be masked instead of 
        fill.
    return_dtype : dtype, optional
        Only valid if is_return_masked_array == True. Data type for the masked array which is returned. 
    Returns:
    --------
    retval : ndarray/masked ndarray
        contains the lowest index in b for each value in a that is a member of b. None/masked if value in a is not
        member of b
    """
    bind = {}
    for i, elt in enumerate(b):
        if (elt not in bind) and (elt is not np.ma.masked):
            bind[elt] = i
    ind = [bind.get(itm, np.nan) for itm in a if a is not np.ma.masked]
    if is_return_masked_array:
        return np.ma.array(np.ma.masked_invalid(ind),dtype=return_dtype)
    else:
        return np.array(ind)

