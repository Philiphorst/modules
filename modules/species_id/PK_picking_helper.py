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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_titles_list_from_dict(range_dict):
    """
    Create a plot_titles list for save_feat_in_pdf() from a range dict
    Parameters:
    -----------
    range_dict : dictionary
        Dictionary whose keys are the filenames and values are list of lists of start and end time
    Returns:
    --------
    plot_titles_list : list    
        A list of strings that are displayed as titles for each subplot when used with save_feat_in_pdf()
    """
    plot_titles_list = []
    for key in range_dict:
        name = key.split('/')[-1]
        for interval in range_dict[key]:
            plot_titles_list.append("{:s}\n{:.2f}-{:.2f}".format(name,interval[0],interval[1]))
    return plot_titles_list
    
    
def save_feat_in_pdf(X,reshape_tuple,pdf_path,plot_titles_list = []):
    """
    Saves a plot of all feature to a pdf file for manual quality control.
    Each page contains 12 plots in 4 rows and 3 columns
    Parameter:
    ----------
    X : ndarray
        Array containing the data. Every row will for one plot
    reshape_tuple : tuple
        Tuple of two integers used to reshape each row of X to create a 2d array
        to be plotted via imshow    
    pdf_path : string
        The full path to which the resulting pdf is saved
    plot_titles_list : list, optional
        A list of strings of the length of X.shape[0] that are displayed as titles for each subplot
    Returns:
    --------
    None
    
    """
    #X=np.zeros((100,10000))
    # -- open a multi page pdf file
    print X.shape
    is_ext_titles = False
    if len(plot_titles_list) == X.shape[0]:
        is_ext_titles = True
    
    with PdfPages(pdf_path) as pdf:
        # -- iterate over all samples (rows in X)
        for i,sample in enumerate(X):
            print i
            # -- position iterator for 12 subplots per page
            pos_it = int(np.mod(i,12))
            # -- if we are at a twelfth figure
            if pos_it == 0:
                fig = plt.figure(figsize=(10,14))
            ax = fig.add_subplot(4,3,pos_it+1)
            ax.imshow(sample.reshape(reshape_tuple)[range(20,60)],origin='lower',aspect=4.)
            if is_ext_titles:
                ax.set_title(plot_titles_list[i],fontsize = 10)
            else:
                ax.set_title('row #{:d}'.format(i))
            if pos_it == 12 - 1 or i == X.shape[0]-1:
                pdf.savefig()
                plt.close()




              
                
if __name__ == '__main__':
    X = np.load('/home/philip/Desktop/tmp/test_data_pec_id/X.npy')
    save_feat_in_pdf(X,(100,100),'/home/philip/Desktop/tmp/test_data_pec_id/X.pdf')  