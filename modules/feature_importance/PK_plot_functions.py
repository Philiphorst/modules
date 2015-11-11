'''
Created on 5 Nov 2015

@author: philip
'''
import numpy as np
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as hierarchy

import modules.feature_importance.PK_ident_top_op as idtop

def plot_arr_dendrogram(abs_corr_array,names):
    """
    Compute  dendrogram and create a plot plotting dendrogram and abs_corr_array
    Parameters:
    ----------
    abs_corr_array : ndarray
        array containing the correlation matrix
    names : list 
        list of strings containing the names of the operations in abs_corr_array in the
        corresponding order.
    Returns:
    --------
    index : list
        list of indices used to reorder the correlation matrix
    """
    # Compute and plot dendrogram.
    fig = plt.figure(figsize=(15,15))
    axdendro = fig.add_axes([0.25,0.76,0.65,0.23])
    corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
    corr_dendrogram = hierarchy.dendrogram(corr_linkage, orientation='top')
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.25,0.1,0.65,0.65])
    index = corr_dendrogram['leaves']
    abs_corr_array = abs_corr_array[index,:]
    abs_corr_array = abs_corr_array[:,index]
    im = axmatrix.matshow(abs_corr_array, aspect='auto', origin='upper',vmin=0,vmax=1)
    axmatrix.set_xticks([])
    axmatrix.set_yticks(range(len(index)))
    axmatrix.set_yticklabels(np.array(names)[index])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.65])
    plt.colorbar(im, cax=axcolor) 
    fig.savefig('/home/philip/work/reports/feature_importance/data/correlation_plots/problem_space/dendr_{:d}_noNorm.png'.format(len(index)))

    return index