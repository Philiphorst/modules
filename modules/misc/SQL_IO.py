'''
Created on 23 Nov 2015

@author: james mccormac
------------------------------------------------------------------------------
Copyright (C) 2015, James McCormac <jmmccormac@gmail.com>,

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
California, 94041, USA.
------------------------------------------------------------------------------
'''
import MySQLdb as mdb
import numpy as np
import numpy.ma as ma

def read_from_mat_file(ts_lower_id, ts_upper_id):
    """
    read hctsa data from mysql database to python data structures
    assumes you want to import a continuous range of ts_ids 

    Parameters:
    ----------
    ts_lower_id : int, the lower ts_id of your timeseries set
    ts_upper_id : int, the upper ts_id of your timeseries set (non-inclusive)
    
    Returns:
    -------
    retval : tuple, tuple of the imported values in the order, timeseries,
    operations, TS_DataMat

    """
    retval = tuple()
    con = mdb.connect('macomp00.ma.ic.ac.uk','jmm09','Upachdoom','jmm09_db')

    cur = con.cursor()

    timeseries = dict()
    for databasekey, pythonkey in zip(['ts_id','Name','Keywords','Length'],['id','filename','keywords','n_samples']):
        cur.execute("SELECT {0} FROM TimeSeries ".format(databasekey) + 
                " WHERE ts_id >= {0} AND ts_id < {1}".format(ts_lower_id,ts_upper_id) +
                " ORDER BY ts_id") 
        ret = np.squeeze(cur.fetchall())
        timeseries[pythonkey] = ret
    retval = retval + (timeseries,)
     
    operations = dict()
    for databasekey,pythonkey in zip(['op_id','Name' ,'Keywords','Code','mop_id'],['id','name','keywords','code_string','master_id']):
        cur.execute("SELECT {0} FROM Operations".format(databasekey) +
                    " ORDER BY op_id")
        ret = np.squeeze(cur.fetchall())
        operations[pythonkey] = ret
    retval = retval + (operations,)

    datamat = []
    for ts_id in range(ts_lower_id,ts_upper_id):
        cur.execute("SELECT Output FROM Results" +
                    " WHERE ts_id = {0}".format(ts_id) +
                    " ORDER BY op_id ")
        ret = np.squeeze(cur.fetchall())

       # This is temporary to check compatibility
       # datamat.append(ret)
    #datamat = np.array(datamat)
    #retval = retval + (datamat,)

       # This code gives the full masked array version  
        cur.execute("SELECT QualityCode FROM Results" +
                    " WHERE ts_id = {0}".format(ts_id) +
                    " ORDER BY op_id ")
        mask = np.squeeze(cur.fetchall())
        masked = ma.masked_array(ret, mask=mask)
        datamat.append(ma.masked_array(ret, mask=mask)) 
    datamat = ma.array(datamat)
    retval = retval + (datamat,)
    
    return retval

if __name__ == '__main__':
    print read_from_mat_file(25367,25375)[2]
