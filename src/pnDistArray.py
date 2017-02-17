#!/usr/bin/env python

"""
Distributed arrays
"""

# standard imports
import functools

# external dependencies
import numpy
from mpi4py import MPI


def distArrayFactory(BaseClass):
    """
    Returns a distributed array class that derives from BaseClass
    @param BaseClass base class, e.g. numpy.ndarray or numpy.ma.masked_array
    @return dist array class
    """

    class DistArrayAny(BaseClass):
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

            # default communicator
            self.comm = MPI.COMM_WORLD

            # winID: {'slice': slce,
            #         'dataSrc': dataSrc,
            #         'dataDst': dataDst,
            #         'window': window}
            self.windows = {}

            # the type of data
            self.dtyp = dtyp

            # this process's MPI rank
            self.rk = self.comm.Get_rank()

            # number of processes
            self.sz = self.comm.Get_size()

            # mapping of numpy data types to MPI data types,
            # assumes that the data are of some numpy variant
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
            elif dtyp == numpy.bool_:
                self.dtypeMPI = MPI.BYTE
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

        def getMPIRank(self):
            """
            Get the MPI rank of this process
            @return rank
            """
            return self.rk

        def getMPISize(self):
            """
            Get the MPI size (number of processes) of this communicator
            @return rank
            """
            return self.sz

        def expose(self, slce, winID):
            """
            Collective operation to expose a sub-set of data
            @param slce tuple of slice objects
            @param winID the data window ID
            """
            # buffer for source data
            dataSrc = numpy.zeros(self[slce].shape, self.dtyp)
            # buffer for destination data
            dataDst = numpy.zeros(self[slce].shape, self.dtyp)

            self.windows[winID] = {
                'slice': slce,
                'dataSrc': dataSrc,
                'dataDst': dataDst,
                'dataWindow': MPI.Win.Create(dataSrc, comm=self.comm),
                }

            if hasattr(self, 'mask'):
                maskSrc = numpy.ones(self[slce].shape, numpy.bool_)
                maskDst = numpy.ones(self[slce].shape, numpy.bool_)
                iw = self.windows[winID]
                iw['maskSrc'] = maskSrc
                iw['maskDst'] = maskDst
                iw['maskWindow'] = MPI.Win.Create(maskSrc, comm=self.comm)

        def getMask(self, pe, winID):
            """
            Access remote mask (collective operation)
            @param pe remote processing element, if None then no operation
            @param winID remote window
            @return mask array or None (if there is no mask)
            @note this is a no operation if there is no mask
            attached to the data
            """
            if 'maskWindow' not in self.windows:
                # no mask, no op
                return None

            iw = self.windows[winID]
            slce = iw['slice']
            maskSrc = iw['maskSrc']
            maskDst = iw['maskDst']

            # copy src mask into buffer
            maskSrc[...] = self.mask[slce]

            win = iw['maskWindow']
            win.Fence(MPI.MODE_NOPUT | MPI.MODE_NOPRECEDE)
            if pe is not None:
                win.Get([maskDst, MPI.BYTE], pe)
            win.Fence(MPI.MODE_NOSUCCEED)

            return maskDst

        def getData(self, pe, winID):
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

            win = iw['dataWindow']
            win.Fence(MPI.MODE_NOPRECEDE) #MPI.MODE_NOPUT | MPI.MODE_NOPRECEDE)

            if pe is not None:
                win.Get([dataDst, self.dtypMPI], pe)

            win.Fence(MPI.MODE_NOSUCCEED)

            return dataDst

        def free(self):
            """
            Must be called to free all exposed windows
            """
            for iw in self.windows:
                self.windows[iw]['dataWindow'].Free()

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

    return DistArrayAny

# create different dist array class flavors
DistArray = distArrayFactory(numpy.ndarray)
MaskedDistArray = distArrayFactory(numpy.ma.masked_array)


#
# Distributed array static constructors
#
def daArray(arry, dtype=numpy.float):
    """
    Array constructor for numpy distributed array
    @param arry numpy-like array
    """
    a = numpy.array(arry, dtype)
    res = DistArray(a.shape, a.dtype)
    res[:] = a
    return res


def daZeros(shap, dtype=numpy.float):
    """
    Zero constructor for numpy distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    """
    res = DistArray(shap, dtype)
    res[:] = 0
    return res


def daOnes(shap, dtype=numpy.float):
    """
    One constructor for numpy distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    """
    res = DistArray(shap, dtype)
    res[:] = 1
    return res


#
# Masked distributed array static constructors
#
def mdaArray(arry, dtype=numpy.float, mask=None):
    """
    Array constructor for masked distributed array
    @param arry numpy-like array
    @param mask mask array (or None if all data elements are valid)
    """
    a = numpy.array(arry, dtype)
    res = MaskedDistArray(a.shape, a.dtype)
    res[:] = a
    res.mask = mask
    return res


def mdaZeros(shap, dtype=numpy.float, mask=None):
    """
    Zero constructor for masked distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    @param mask mask array (or None if all data elements are valid)
    """
    res = MaskedDistArray(shap, dtype)
    res[:] = 0
    res.mask = mask
    return res


def mdaOnes(shap, dtype=numpy.float, mask=None):
    """
    One constructor for masked distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    @param mask mask array (or None if all data elements are valid)
    """
    res = MaskedDistArray(shap, dtype)
    res[:] = 1
    res.mask = mask
    return res


######################################################################

def test():

    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    sz = comm.Get_size()

    # create local data container
    n, m = 3, 4
    data = numpy.reshape(numpy.array([rk*100.0 + i for i in range(n*m)]),
                         (n, m))

    # create dist array
    da = DistArray(data.shape, data.dtype)

    # load the data
    da[:] = data

    # expose data to other pes
    da.expose((slice(None, None,), slice(-1, None,)), winID='east')

    # fetch data
    if rk > 0:
        daOtherEast = da.getData(pe=rk-1, winID='east')
    else:
        daOtherEast = da.getData(pe=sz-1, winID='east')

    # check
    daLocalEast = da[da.windows['east']['slice']]
    diff = daLocalEast - daOtherEast
    if rk > 0:
        try:
            assert(numpy.all(diff == 100))
            print('[{0}]...OK'.format(rk))
        except:
            sLocal = str(daLocalEast)
            sOther = str(daOtherEast)
            print('[{0}] daLocalEast={1}\ndaOtherEast={2}'.format(
                                                                  rk,
                                                                  sLocal,
                                                                  sOther))
            print('error: {0}'.format(numpy.sum(diff - 100)))
    else:
        try:
            assert(numpy.all(diff == -100*(sz-1)))
            print('[{0}]...OK'.format(rk))
        except:
            sLocal = str(daLocalEast)
            sOther = str(daOtherEast)
            print('[{0}] daLocalEast={1}\ndaOtherEast={2}'.format(rk,
                                                                  sLocal,
                                                                  sOther))
            print('error: {0}'.format(numpy.sum(diff + 100*(sz-1))))

    # delete windows
    da.free()

if __name__ == '__main__':
    test()
