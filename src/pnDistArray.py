#/usr/bin/env python

"""
Distributed array class

This code is provided with the hope that it will be useful. 
No guarantee is provided whatsoever. Use at your own risk.

See LICENSE.txt for conditions of use. 

Alex Pletzer (2012)
"""

__version__ = "$Id: pnDistArray.py 49 2016-05-27 11:08:50Z pletzer $"

# standard imports
import copy
import functools

# external dependencies
import numpy
from mpi4py import MPI
from functools import reduce

class DistArray(numpy.ndarray):
    """
    Distributed array. Each process owns data and can expose a subset 
    of the data to other processes. These are known as windows. Any 
    number of windows can be exposed and the data of windows can be 
    overlapping. Any process can access exposed windows from any other
    process. This relies on MPI-2 one-sided get communication.
    """

    def __new__(cls, *args, **kwargs):
        return numpy.ndarray.__new__(cls, *args, **kwargs)

    def __init__(self, shap, dtyp):
        """
        Constructor
        @param shap array shape
        @param dtyp numpy type
        """

        self.comm = MPI.COMM_WORLD # default communicator
        self.windows = {}          # winID: {'slice': slce,
                                   #         'dataSrc': dataSrc,
                                   #         'dataDst': dataDst,
                                   #         'window': window}

        self.dtyp = dtyp
        self.rk = self.comm.Get_rank()
        self.sz = self.comm.Get_size()

        self.dtypMPI = None
        if dtyp == numpy.float64:
            self.dtypMPI = MPI.DOUBLE
        elif dtyp == numpy.float32:
            self.dtypeMPI = MPI.FLOAT
        elif dtyp == numpy.int64:
            self.dtypeMPI = MPI.INT64_T
        elif dtyp == numpy.int32:
            self.dtypeMPI = MPI.INT32_T
        elif dtyp == numpy.int16:
            self.dtypeMPI = MPI.INT16_T
        elif dtyp == numpy.int8:
            self.dtypeMPI = MPI.INT8_T
        else:
            raise NotImplementedError

    def setComm(self, comm):
        """
        Set communicator
        @param comm communicator
        """
        self.comm = comm
        self.rk = self.comm.Get_rank()
        self.sz = self.comm.Get_size()

    def expose(self, slce, winID):
        """
        Collective operation to expose a sub-set of data
        @param slce tuple of slice objects
        @param winID the data window ID
        """
        # buffer for source data
        dataSrc = numpy.ones(self[slce].shape, self.dtyp) 
        # buffer for destination data
        dataDst = numpy.ones(self[slce].shape, self.dtyp)
        self.windows[winID] = {
            'slice': slce,
            'dataSrc': dataSrc,
            'dataDst': dataDst,
            'window': MPI.Win.Create(dataSrc, comm=self.comm),
            }

    def get(self, pe, winID):
        """
        Access remote data (collective operation)
        @param pe remote processing element, if None then no operation
        @param winID remote window
        @return array
        """
        iw = self.windows[winID]
        slce = iw['slice']
        dataSrc = iw['dataSrc']
        dataDst = iw['dataDst']

        # copy src data into buffer
        dataSrc[...] = self[slce]

        win = iw['window']
        win.Fence()
        if pe is not None:
            win.Get( [dataDst, self.dtypMPI], pe )
        win.Fence()

        return dataDst

    def free(self):
        """
        Must be called to free all exposed windows
        """
        for iw in self.windows:
            self.windows[iw]['window'].Free()

    def reduce(self, op, initializer=None, rootPe=0):
        """
        Collective reduce operation
        @param op function (e.g. lambda x,y:x+y)
        @param initializer the return value if there are no elements
        @param rootPe the root process which receives the result
        @return result on rootPe, None on all other processes
        """
        if len(self) == 0:
            return initializer
        val = functools.reduce(op, self.flat)
        data = self.comm.gather(val, rootPe)
        if self.rk == rootPe:
            return functools.reduce(op, data)
        else:
            return None

def daArray(arry, dtype=numpy.float):
    """
    Array constructor
    @param arry numpy-like array
    """
    a = numpy.array(arry, dtype)
    res = DistArray(a.shape, a.dtype)
    res[:] = a # copy
    return res

def daZeros(shap, dtype=numpy.float):
    """
    Zero constructor
    @param shap the shape of the array
    @param dtype the numpy data type 
    """
    res = DistArray(shap, dtype)
    res[:] = 0
    return res

def daOnes(shap, dtype=numpy.float):
    """
    One constructor
    @param shap the shape of the array
    @param dtype the numpy data type 
    """
    res = DistArray(shap, dtype)
    res[:] = 1
    return res

######################################################################

def test():

    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    sz = comm.Get_size()
    
    # create local data container
    n, m = 3, 4
    data = numpy.reshape(numpy.array( [rk*100.0 + i for i in range(n*m) ] ), (n,m))

    # create dist array
    da = DistArray(data.shape, data.dtype)
    
    # load the data
    da[:] = data

    # expose data to other pes
    da.expose( (slice(None, None,), slice(-1, None,)), winID='east' )

    # fetch data
    if rk > 0:
        daOtherEast = da.get(pe=rk-1, winID='east')
    else:
        daOtherEast = da.get(pe=sz-1, winID='east')

    # check
    daLocalEast = da[ da.windows['east']['slice'] ]
    diff = daLocalEast - daOtherEast
    if rk > 0:
        try:
            assert( numpy.all( diff == 100 ) )
            print('[{0}]...OK'.format(rk))
        except: 
            print('[{0}] daLocalEast={1}\ndaOtherEast={2}'.format(rk, str(daLocalEast), str(daOtherEast)))
            print('error: {0}'.format(numpy.sum(diff - 100)))
    else:
        try:
            assert( numpy.all( diff == -100*(sz-1) ) )
            print('[{0}]...OK'.format(rk))
        except:
            print('[{0}] daLocalEast={1}\ndaOtherEast={2}'.format(rk, str(daLocalEast), str(daOtherEast)))
            print('error: {0}'.format(numpy.sum( diff + 100*(sz-1) )))

    # delete windows
    da.free()

if __name__ == '__main__': test()
