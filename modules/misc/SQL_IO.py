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

#Database information
host = 'macomp00.ma.ic.ac.uk'
user = 'jmm09'
password = 'Upachdoom'
database = 'jmm09_db'

def read_from_sql(ts_ids, hctsa_struct_names):
    """
    read hctsa data from a mysql database to python data structures

    Parameters:
    ----------
    ts_ids : list, a list of the ts_ids you want to import 
    hctsa_struct_names : list
        List of strings of identifiers for which structures are to
        be read from the mat_file. 
        Possible values are : 'TimeSeries','Operations','TS_DataMat'

    Returns:
    -------
    retval : tuple, tuple of the imported values in the order, timeseries,
    operations, TS_DataMat

    """

    ts_ids = tuple(ts_ids)
    retval = tuple()
    con = mdb.connect(host,user,password,database)

    cur = con.cursor()

    for item in hctsa_struct_names:
        if item == 'TimeSeries':
            timeseries = dict()
            for databasekey, pythonkey in zip(['ts_id','Name','Keywords','Length'],
                                      ['id','filename','keywords','n_samples']):
                cur.execute("SELECT {0} FROM TimeSeries ".format(databasekey) + 
                            " WHERE ts_id IN {0}".format(ts_ids) +
                            " ORDER BY ts_id") 
                ret = np.squeeze(cur.fetchall()).tolist()
                timeseries[pythonkey] = ret
            retval = retval + (timeseries,)
     
        if item == 'Operations':
            operations = dict()
            for databasekey,pythonkey in zip(['op_id','Name' ,'Keywords','Code','mop_id'],
                                     ['id','name','keywords','code_string','master_id']):
                cur.execute("SELECT {0} FROM Operations".format(databasekey) +
                    " ORDER BY op_id")
                ret = np.squeeze(cur.fetchall()).tolist()
                operations[pythonkey] = ret
            retval = retval + (operations,)

        if item == 'TS_DataMat':
            datamat = []
            for ts_id in ts_ids:
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

    con.close()
    
    return retval

def read_from_sql_by_filename(filenamelike, hctsa_struct_names):
    """
    A wrapper of read_from_sql which accepts a regular expression
    to specify type of name

    Parameters
    ----------
    filenamelike: string, regular expression with wildcard % 
    
    Returns:
    -------
    tuple, tuple of the imported values in the order, timeseries,
    operations, TS_DataMat
    """
    con = mdb.connect(host,user,password,database)

    cur = con.cursor()
    
    cur.execute("SELECT ts_id FROM TimeSeries" +
                " WHERE Name LIKE '{0}'".format(filenamelike))
    
    ret = np.squeeze(cur.fetchall())
    
    con.close()
    return read_from_sql(ret, hctsa_struct_names)

if __name__ == '__main__':
    print read_from_sql_by_filename('Coffee_%',
                                    ['TimeSeries','Operations', 'TS_DataMat'])
