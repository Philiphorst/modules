'''
Created on 5 Nov 2015

@author: philip
'''
import numpy as np
import scipy.spatial.distance as spdst  
import scipy.cluster.hierarchy as hierarchy

def calc_linkage(abs_corr_array, linkage_method='complete'):  
    """
    Calculate the linkage for a absolute correlation array
    Parameters:
    -----------
    abs_corr_array : ndarray
        array containing the correlation matrix
    linkage_method : str,optional
        The linkage algorithm to use. 
    Returns:
    --------
    link_arr : ndarray
        The hierarchical clustering encoded as a linkage matrix.
    abs_corr_dist_arr : ndarray
        The distance array calculated from abs_corr_array
    """
    # -- transform the correlation matrix into distance measure
    abs_corr_dist_arr = np.around(1 - abs_corr_array,7)
    # -- XXX The correlation sometime is larger than 1. not sure if that is a negligible or has to
    # -- be sorted out
    abs_corr_dist_arr[(abs_corr_dist_arr < 0)] = 0

    # -- transform the correlation matrix into condensed distance matrix
    dist_corr = spdst.squareform(abs_corr_dist_arr)  

    # -- calculate the linkage
    link_arr = hierarchy.linkage(dist_corr, method=linkage_method)
    
    return link_arr,abs_corr_dist_arr

def calc_perform_corr_mat(all_classes_avg_good,IS_SORT_NORMALISED = False, max_feat = 200):
    """
    Calculate the correlation matrix of the performance for top features
    Parameters:
    -----------
    all_classes_avg_good : masked ndarray
        Masked array containing the average statistics for each problem(row = problem) 
        for each operation (column = operation)
    IS_SORT_NORMALISED : bool
        If True, the order of top features is calculated by first normalising all features per problem.
        Otherwise no normalisation is applied
    max_feat : int
        Max number of feature for which the correlation is calculated
    Retruns:
    --------
    abs_corr_array : nd_array
        Array containing the correlation matrix for the top max_feat features
    """
      
    if IS_SORT_NORMALISED:
        all_classes_avg_good_mean = np.ma.masked_invalid(np.ma.mean(all_classes_avg_good,axis = 1))
        all_classes_avg_good = (all_classes_avg_good.T / all_classes_avg_good_mean).T
    sort_ind = np.argsort(all_classes_avg_good.sum(axis=0))
    acag_n_sort_red = all_classes_avg_good[:,sort_ind[:max_feat]] 
    
    # -- calculate the correlation
    abs_corr_array = np.abs(np.ma.corrcoef(acag_n_sort_red, rowvar=0))      
    return abs_corr_array,sort_ind



