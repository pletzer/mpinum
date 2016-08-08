import pnumpy
import numpy
import unittest
from mpi4py import MPI
from functools import reduce

class TestDistArray(unittest.TestCase):
    """
    Test pnumpy
    """

    def setUp(self):
        pass

    def test0(self):
        """
        Test constructors
        """
        da = pnumpy.daZeros( (2,3), numpy.float64 )
        da = pnumpy.daOnes( (2,3), numpy.float64 )
        da = pnumpy.daArray( [1,2,3] )

    def test1d_1(self):
        """
        1D, float64
        """

        dtyp = numpy.float64

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        da = pnumpy.daZeros( (n,), dtyp )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc {0} tries to access data from {1}'.format(rk, leftRk))
        leftData = da.get(pe=leftRk, winID='left')
        print('leftData for rank {0} = {1}'.format(rk, str(leftData)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test1d_2(self):
        """
        1D, float32
        """

        dtyp = numpy.float32

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        da = pnumpy.daZeros( (n,), dtyp )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc %d tries to access data from %d' % (rk, leftRk))
        leftData = da.get(pe=leftRk, winID='left')
        print('leftData for rank {0} = {1}'.format(rk, str(leftData)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test1d_3(self):
        """
        1D, int
        """

        dtyp = numpy.int64

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        da = pnumpy.daZeros( (n,), dtyp )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc %d tries to access data from %d' % (rk, leftRk))
        leftData = da.get(pe=leftRk, winID='left')
        print('leftData for rank %d = %s' % (rk, str(leftData)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test1d_4(self):
        """
        1D, bool
        """

        dtyp = numpy.bool_

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        da = pnumpy.daZeros( (n,), dtyp )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = [(i%2 == 0) for i in range(n)]
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc %d tries to access data from %d' % (rk, leftRk))
        leftData = da.get(pe=leftRk, winID='left')
        print('leftData for rank %d = %s' % (rk, str(leftData)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test2d_1(self):
        

        # create the dist array, the sizes are local to each processor
        da = pnumpy.daZeros( (2,3), numpy.float32 )

        # processor rank and number of processes
        rk = da.rk
        nprocs = da.sz

        # expose sub-domains
        northSlab = ( slice(-1, None, None), slice(0, None, None) )
        da.expose( slce=northSlab, winID='n' )

        # set the data 
        da[:] = rk

        # access the remote data, collective operation
        northData = da.get( (rk-1) % nprocs, winID='n' )

        # check sum
        checkSum = da.reduce(lambda x,y:x+y, rootPe=0)
        if checkSum is not None:
            exactCheckSum = reduce(lambda x,y:x+y,
                                   [rank for rank in range(nprocs)])
            exactCheckSum *= len(da.flat)
            self.assertEqual(checkSum, exactCheckSum) 

        # check 
        self.assertEqual(northData.min(), (rk - 1) % nprocs)
        self.assertEqual(northData.max(), (rk - 1) % nprocs)

        # clean up
        da.free()
       
if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDistArray)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
    
