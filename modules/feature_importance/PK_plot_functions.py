'''
Created on 5 Nov 2015

@author: philip
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as hierarchy

import modules.feature_importance.PK_ident_top_op as idtop
def plot_arr_dendrogram_bak(abs_corr_array,names):
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
    figsize=(10,10)
    rect_dendro = [0.25,0.76,0.65,0.23]
    rect_matrix = [0.25,0.1,0.65,0.65]
    rect_color = [0.91,0.1,0.02,0.65]
    
    
    # Compute and plot dendrogram.
    fig = plt.figure(figsize=figsize)
    axdendro = fig.add_axes(rect_dendro)
    corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
    corr_dendrogram = hierarchy.dendrogram(corr_linkage, orientation='top')
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes(rect_matrix)
    index = corr_dendrogram['leaves']
    abs_corr_array = abs_corr_array[index,:]
    abs_corr_array = abs_corr_array[:,index]
    im = axmatrix.matshow(abs_corr_array, aspect='auto', origin='upper',vmin=0,vmax=1)
    axmatrix.set_xticks([])
    axmatrix.set_yticks(range(len(index)))
    axmatrix.set_yticklabels(np.array(names)[index])

    # Plot colorbar.
    axcolor = fig.add_axes(rect_color)
    plt.colorbar(im, cax=axcolor) 
    #fig.savefig('/home/philip/work/reports/feature_importance/data/correlation_plots/problem_space/dendr_{:d}_Norm.png'.format(len(index)))

    return index

def plot_arr_dendrogram(abs_corr_array,names,max_dist_cluster,measures = None):
    """
    Compute  dendrogram and create a plot plotting dendrogram and abs_corr_array
    Parameters:
    ----------
    abs_corr_array : ndarray
        array containing the correlation matrix
    names : list 
        list of strings containing the names of the operations in abs_corr_array in the
        corresponding order.
    max_dist_cluster : float
        Maximum distance in the clusters
    measures : ndarray (n_measures x abs_corr_array.shape[0])
        Array containing measures to be plotted on top of the matrix. Positions corresponding positions
        of operations in abs_corr_array.
    Returns:
    --------
    index : list
        list of indices used to reorder the correlation matrix
    """
 
    figsize=(18,12)    
    #figsize=(46.81,33.11) 
    rect_measures = [0.25,0.8075,0.5,0.15]
    rect_dendro = [0.755,0.05,0.15,0.75]
    rect_matrix = [0.25,0.05,0.5,0.75]
    rect_color = [0.92,0.05,0.02,0.75]
    

    # Compute and plot dendrogram.
    fig = plt.figure(figsize=figsize)
    axdendro = fig.add_axes(rect_dendro)
    corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
      
    corr_dendrogram = hierarchy.dendrogram(corr_linkage, orientation='left')
    #axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axvline(max_dist_cluster,ls='--',c='k')
    # Plot distance matrix.
    axmatrix = fig.add_axes(rect_matrix)
    index = corr_dendrogram['leaves']
    abs_corr_array = abs_corr_array[index,:]
    abs_corr_array = abs_corr_array[:,index]
    
    # -- plot the correlation matrix
    im = axmatrix.matshow(abs_corr_array, aspect='auto', origin='lower',vmin=0,vmax=1)
      
    axmatrix.set_xticks([])
    axmatrix.set_yticks(range(len(index)))
    #axmatrix.set_yticklabels(np.array(names)[index],fontsize=5)
    axmatrix.set_yticklabels(np.array(names)[index])

    # Plot colorbar.
    axcolor = fig.add_axes(rect_color)
    plt.colorbar(im, cax=axcolor) 
    
    
    # Plot the quality measures
    axmeasure = fig.add_axes(rect_measures)
    axmeasure.xaxis.set_ticklabels([]) 
    axmeasure.scatter(np.arange(0,measures.shape[-1])+0.5,measures[0,index])
    axmeasure.set_xlim([0,measures.shape[-1]])
    axmeasure.set_ylabel('problems calculated')
    axmeasure.yaxis.label.set_color('b')
    [label.set_color('b') for label in axmeasure.get_yticklabels()]
    axmeasure2 = axmeasure.twinx()
    axmeasure2.plot(np.arange(0,measures.shape[-1])+0.5,measures[1,index],color='r')
    axmeasure2.set_xlim([0,measures.shape[-1]])

    [label.set_color('r') for label in axmeasure2.get_yticklabels()]
    axmeasure2.set_ylabel('z-scored avg u-stat')
    axmeasure2.yaxis.label.set_color('r')

    # -----------------------------------------------------------------
    # -- calculate and plot clusters ----------------------------------
    # -----------------------------------------------------------------
    #cluster_ind = hierarchy.fcluster(link_arr, t=cluster_t, criterion=cluster_criterion)
    cluster_ind = hierarchy.fcluster(corr_linkage, t = max_dist_cluster, criterion='distance')
                                     
    # -- plot delimiters for measures
    cluster_bounds = np.hstack((-1,np.nonzero(np.diff(cluster_ind[index]))[0],abs_corr_array.shape[0]-1))+1
    for bound in cluster_bounds:
        axmeasure.axvline(bound,linestyle='--',color='k')                            
                                     
    # -- calculate the locations for the cluster squares
    patch_bounds = cluster_bounds - .5
    patch_sizes = np.diff(patch_bounds)
    cluter_square_params = tuple(((patch_bounds[i],patch_bounds[i]),patch_sizes[i],patch_sizes[i]) for i in range(len(patch_sizes)))
    for cluster_square_param in cluter_square_params:
        axmatrix.add_patch(mpl.patches.Rectangle(cluster_square_param[0],cluster_square_param[1],cluster_square_param[2],fill=0,ec='w',lw=2))  

    
    # -----------------------------------------------------------------
    # -- calculate and plot best features -----------------------------
    # -----------------------------------------------------------------  
    best_features_marker = []
    for (i,j) in zip(cluster_bounds[:-1],cluster_bounds[1:]):
        measures_dendr = measures[1,index]
        best_features_marker.append(i+np.argmin(measures_dendr[i:j]))
        
    axmatrix.scatter(best_features_marker,best_features_marker,color='w') 
    axmatrix.set_xlim([-0.5,abs_corr_array.shape[0]-0.5])
    axmatrix.set_ylim([-0.5,abs_corr_array.shape[0]-0.5])
    
    [(text.set_color('k'),text.set_weight('bold')) for i,text in enumerate(axmatrix.get_yticklabels()) if i in best_features_marker]
    
    
    
    return index






if __name__ == "__main__":
    pass
    
    
    