'''
Created on 23 Jun 2015

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
import glob 
import numpy as np
import modules.misc.PK_matlab_IO as mIO
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_feature_quality as fq
import scipy.spatial.distance as spdst
import scipy.cluster.hierarchy as hierarchy

def cat_data_from_matfile_root(mat_file_paths, count_op_id_min,is_from_old_matlab = False,
                               is_return_masked = True, data_all_good_op_path = './',
                               op_id_good_path = './'):
    """
    Concatenate all data arrays for all HCTSA_loc.mat files located in mat_file_root. Only keep features that 
    have been successfully calculated for count_op_id_min problems. Saves data_all and op_id_good to disk
    Parameters:
    -----------
    mat_file_paths : list
        List containing path to all relevant MAT files.
    count_op_id_min : int
        Number of problems for which each feature has to be calculated to be included
    is_from_old_matlab : boolean
        Are the MAT files from older version of the comp engine
    is_return_masked : boolean
        Saving large masked arrays to disk can lead to memory errors while pickling. If this is false funtion
        returns a normal ndarray with unknown entires are set to NaN. This can be converted to a masked array with 
        data_all = np.ma.masked_invalid(data_all)
    data_all_good_op_path,op_id_good_path : string
        Output paths for data_all and op_id_good respectively.
        
    Returns:
    --------
    data_all : ndarray, masked array
        Array containing data from all problems, where each row contains the good features of one timeseries.
    op_id_good : ndaray
        Op id's of features that have been calculated in at least count_op_id_min problems.
    """
    # -- calculate in how many problems each operation has been successfully calculated
    count_op_id = fq.count_op_calc(mat_file_paths,is_from_old_matlab = is_from_old_matlab)    
    
    # -- pick only features calculated in all problems
    op_id_good = np.nonzero([count_op_id >= count_op_id_min])[1]
     
    # -- concatenate good features for all problems to one large feature matrix (can take a while and get big)
    data_all = cat_data_op_subset(mat_file_paths,op_id_good,is_from_old_matlab = is_from_old_matlab,is_return_masked = is_return_masked)

    # -- safe the calculated values
    if is_return_masked == True:
        np.ma.dump(data_all, data_all_good_op_path)
    else:
        np.save(data_all_good_op_path,data_all)
        
    np.save(op_id_good_path,op_id_good)
    
    return data_all,op_id_good
 
def cat_data_op_subset(file_paths,op_id_top,is_from_old_matlab = False,is_return_masked = True):
    """
    Concatenate the features where op_id is in op_id_top for all HCTSA_loc.m files pointed to by file_paths.
    Warning, this can take a while and the returned data matrix can be very large.
    XXX WARNING XXX This only works correctly if all HCTSA_loc.mat files come from the same
    database. Meaning op_ids are the same. Otherwise one would have to go through operation names which is
    only a little more work to implement. XXX
    Parameters:
    -----------
    file_paths : list
        list of file paths pointing to the files containing the data
    op_id_top : list,ndarray
        list of operation ids wanted in the concatenated data array
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.
    is_return_masked : boolean
        Saving large masked arrays to disk can lead to memory errors while pickling. If this is false funtion
         returns a normal ndarray with unknown entires are set to NaN. This can be converted to a masked array with 
         data_all = np.ma.masked_invalid(data_all)
    Returns:
    --------
    data_all : ndarray/masked ndarray
        Concatenated data array
   """
    is_first = True
    data_all = None

    for file_path in file_paths:
        print "Adding data from {:s} \n to complete data matrix".format(file_path)
        data,op = mIO.read_from_mat_file(file_path, ['TS_DataMat','Operations'],is_from_old_matlab = is_from_old_matlab)
 
        # -- find the indices in the data for for op_id_top
        ind = hlp.ismember(op['id'],op_id_top,is_return_masked_array = True,return_dtype = int)
        # -- if any of the operations was not calculated for this problem
        # -- create a masked array and copy only valid data and mask 
        # -- invalid data
        if ind.data != op_id_top:
            # -- create an masked array filled with NaN. 
            # -- This makes later masking of non-existent entries easier
            # -- each column of data_ma corresponds to the op_id in op_id_top with the
            # -- same index (column i in data_ma corresponds to op_id_top[i])

            data_ma = np.empty((data.shape[0],np.array(op_id_top).shape[0]))
            data_ma[:] = np.NaN
            for it,i in enumerate(ind):
                # -- if i is masked in ind that means that the current operation in data
                # -- is not part of op_id_top. We therefore do not need this operation to 
                # -- be included in data_ma.
                if i is not np.ma.masked:
                    data_ma[:,i] = data[:,it]
        # -- otherwise pick all relevant features and also automatically sort them correctly (if necessary)
        else:
            data_ma = np.array(data[:,ind])
        
        # -- mask all NaN (not calculated) entries and stick them together
        #data_ma = np.ma.masked_invalid(data_ma)
        if is_first == True:
            data_all = data_ma
            is_first = False
        else:
            data_all = np.vstack((data_all,data_ma))
    # -- Saving a large masked array to disk can lead to Memory errors while using the pickle module.
    if is_return_masked == True:
        data_all = np.ma.masked_invalid(data_all)
    return data_all

def compute_clusters_from_dist(abs_corr_dist_arr = None,link_arr_path = None, 
                               link_arr = None, is_force_calc_link_arr = False,
                               cluster_t = .2, linkage_method='complete',
                               cluster_criterion ='distance'):
    
    """ Compute the clustering from a distance matrix or a saved linkage matrix.
    The linkage array will be calculated if is_force_calc_link_arr == True or
    if link_arr == None and link_arr_path != None. Calculation of the link array requires 
    abs_corr_dist_arr != None.
    The linkage array will be loaded from file if is_force_calc_link_arr == False and
    link_arr_path != None. 
        
    
    Parameters:
    -----------
    abs_corr_dist_arr : ndarray,optional
        redundant distance array, containing the distances between all pairs of samples.
        Required if linkage is to be calculated.
    link_arr_path : string, optional
        path to a npy file containing a linkage array.
    link_arr : ndarray, optional
        linkage array
    is_force_calc_link_arr : bool
        is recalculation of the linkage array forced if link_arr_path is given.   
    cluster_t : float
        The threshold to apply when forming flat clusters. See scipy.cluster.hierarchy.fcluster()
    linkage_method : str, optional
        The linkage algorithm to use. See scipy.cluster.hierarchy.linkage()   
    cluster_criterion : str, optional
        The criterion to use in forming flat clusters. See scipy.cluster.hierarchy.fcluster()
    Returns:
    --------
    cluster_lst : list of lists 
        list of lists of indices of the abs_corr_dist_arr given as parameter where every sublist describes a cluster
    cluster_size_lst : list
        sizes of clusters contained in cluster_lst
     """

    
    # -- calculate new linkage no matter what
    if is_force_calc_link_arr:
        # -- transform the correlation matrix to vector form distance
        dist_corr = spdst.squareform(abs_corr_dist_arr)
        link_arr = hierarchy.linkage(dist_corr, method=linkage_method)
        # -- if a path to a link_array is given, override the array
        if link_arr_path != None:
            np.save(link_arr_path, link_arr)
    # -- only calculate new linkage if necessary
    else:
        # -- if a path to a link_array is given, load the array
        if link_arr_path != None:
            link_arr = np.load(link_arr_path)
        # -- otherwise recalculate
        else:
            # -- transform the correlation matrix to vector form distance
            dist_corr = spdst.squareform(abs_corr_dist_arr)
            link_arr = hierarchy.linkage(dist_corr, method=linkage_method)
            
    # -- compute clusters
    cluster_ind = hierarchy.fcluster(link_arr, t=cluster_t, criterion=cluster_criterion)
    
    # -- map operations to clusters
    n_cluster = max(cluster_ind)

    # -- create list of list where sublists are clusters and entries are indices in abs_corr_array
    cluster_lst = [[] for _ in xrange(n_cluster)]
    for i in xrange(len(cluster_ind)):
        cluster_lst[cluster_ind[i]-1].append(i)
    # -- size of each cluster in 
    cluster_size_lst = [len(cluster) for cluster in cluster_lst]

    return cluster_lst, cluster_size_lst



def corelated_features_mask(data=None,abs_corr_array=None,calc_times=None,op_ids=None):
    """
    Computes a mask that, when applied, removes correlated features from the data array.
    Parmeters:
    ----------
    data : ndarray
        A data matrix with rows represent training samples and columns represent features
    abs_corr_array : ndarray
        The correlation matrix of all features. Has to be given id data == none
    calc_times : ndarray
        Array where the first row corresponds to operation id's and the second row to calculation times
        for these operation ids
    op_ids : 
        The operation ids corresponding to the rows/columns in abs_corr_array
    Returns:
    -------
    mask : ndarray,dtype=bool
        1d array whose entries are one only for uncorrelated entries.
    abs_corr_arrayabs_corr_array : ndarray
        the correlation matrix
    """
    
    if abs_corr_array == None:
        abs_corr_array = np.abs(np.ma.corrcoef(data, rowvar=0))  
      
    # -- Vector containing 0 for all operations we don't want
    mask = np.ones(abs_corr_array.shape[0],dtype='bool') 
      
    for i in range(abs_corr_array.shape[0]):
    # -- if the current line represents an operation not yet eleminated
        if mask[i]:
            # -- remove operations which are highly correlated
            mask[(abs_corr_array[i] > 0.8)] = 0 
            #----------------------------------------------------------
            # -- Use fastest operation in a set of correlated operations
            #----------------------------------------------------------
            if calc_times != None and op_ids != None:
                # -- find ind in abs_corr_array of correlated features
                ind_corr = np.nonzero((abs_corr_array[i] > 0.8))[0]
    
                # -- translate ind_corr to op ids
                op_id_corr = hlp.ind_map_subset(range(abs_corr_array.shape[0]),op_ids,ind_corr)
    
                # -- get calculation time of correlated op ids
                t_corr = hlp.ind_map_subset(calc_times[0],calc_times[1],op_id_corr)
                # -- check if all entries are None -> no timing information for any of the operations
                if np.nonzero(t_corr)[0].shape[0] == 0:
                    # -- pick the first operation as fastest as there is no timing information
                    op_id_corr_fastest = op_id_corr[0]  
                # -- else pick the fastest operation               
                else:
                    # -- get op_id of fastest operation in this correlated set
                    op_id_corr_fastest = op_id_corr[np.nanargmin(t_corr)]
    
                # -- get index of fastest operation in this correlated set
                ind_corr_fastest = hlp.ind_map_subset(op_ids,range(abs_corr_array.shape[0]),op_id_corr_fastest)
                
                # -- add fastest index back in
                mask[ind_corr_fastest] = 1            
            #----------------------------------------------------------
            # -- Use arbitrary operation in a set of correlated operations
            #----------------------------------------------------------
            else:
                mask[i] = 1
    return mask,abs_corr_array

def normalise_array(data,axis,norm_type = 'zscore'):
    """
    normalise the array along the axis.
    Parameters:
    -----------
    data : ndarray
        The array containing the data to be normalised
    axis : int
        The axis along which data is normalised
    norm_type : string
        Type of normalisation. Can be 'zscore','sigmoid_mean_std','sigmoid'
    Returns:
    --------
    data : ndarray
        The normalised data array
    [parameters] : float
        The parameters calculated for the normalisation. Depending on norm_type
    """
    if norm_type == 'sigmoid_mean_std':
        # sigmoidal transform over learning samples
        # enable broadcasting for both cases of axis       
        mean = np.expand_dims(np.mean(data, axis=axis),axis=axis)
        std = np.expand_dims(np.std(data, axis=axis),axis=axis)

        data = 1/(1+np.exp(-(data-mean)/(std)))
        return (data*2.)-1,mean,std
    
    if norm_type == 'sigmoid':
        # outlier-robust sigmoidal transform over learning samples
        q75, q50, q25 = np.percentile(data, [75 ,50, 25],axis = axis)
        iqr = q75 - q25 
        #print  np.amax(data[0]), data[0].mean(), data[0].var()
        #print  q75[0], q50[0], q25[0] 
        if axis != None:
            # enable broadcasting for both cases of axis
            iqr = np.expand_dims(iqr,axis=axis)
            q50 = np.expand_dims(q50,axis=axis)
        #non_zero_iqr_ind = (iqr != 0.0)       
        data = 1/(1+np.exp(-(data-q50)/(iqr/1.35)))
        return (data*2.)-1,q50,iqr
    
    elif norm_type == 'zscore':
        # enable broadcasting for both cases of axis
        mean = np.expand_dims(np.mean(data, axis=axis),axis=axis)
        std = np.expand_dims(np.std(data, axis=axis),axis=axis)
        #print "mean.shape",mean.shape,"std.shape",std.shape
        # zscore over all features 
        data = ( data - mean ) / std
        return data,mean,std
    
def normalise_masked_array(data,axis,norm_type = 'zscore'):
    """
    normalise the masked array along the axis.
    Parameters:
    -----------
    data : ndarray/masked array
        The array containing the data to be normalised
    axis : int
        The axis along which data is normalised
    norm_type : string
        Type of normalisation. Can be 'zscore','sigmoid'
    Returns:
    --------
    data : ndarray
        The normalised data array
    [parameters] : float
        The parameters calculated for the normalisation. Depending on norm_type
    """
    if norm_type == 'sigmoid_mean_std':
        # sigmoidal transform over learning samples
        # enable broadcasting for both cases of axis       
        mean = np.ma.expand_dims(np.ma.mean(data, axis=axis),axis=axis)
        std = np.ma.expand_dims(np.ma.std(data, axis=axis),axis=axis)

        data = 1/(1+np.exp(-(data-mean)/(std)))
        return (data*2.)-1,mean,std
    
   
    elif norm_type == 'zscore':
        # enable broadcasting for both cases of axis
        mean = np.ma.expand_dims(np.ma.mean(data, axis=axis),axis=axis)
        std = np.ma.expand_dims(np.ma.std(data, axis=axis),axis=axis)
        #print "mean.shape",mean.shape,"std.shape",std.shape
        # zscore over all features 
        data = ( data - mean ) / std

        return data,mean,std
    
    
    
    