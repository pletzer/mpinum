import pnumpy
import numpy
import unittest
from functools import reduce
from mpi4py import MPI

class TestPNumpy(unittest.TestCase):
    """
    Test pnumpy
    """

    def setUp(self):
        pass

    def test1d_1(self):
        """
        1d, float64
        """

        dtyp = numpy.float64

        # create the ghosted dist array
        n = 10
        da = pnumpy.gdaZeros( (n,), dtyp, numGhosts=1 )

        # set data to process dependent value, 
        # da.rk is the mpi proc ID
        # da.sz is the size of the MPI communicator
        da[:] = 100*da.rk + numpy.array([i for i in range(n)], dtyp)

        # access remote data to the left
        leftRk = (da.rk - 1) % da.sz

        print('proc %d tries to access data from %d' % (da.rk, leftRk))
        leftData = da.getData(pe=leftRk, winID=(1,))

        print('leftData for rank %d = %s' % (da.rk, str(leftData)))
        # check
        if leftRk < da.rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(da.sz-1))

        # free
        da.free()

    def test2d_1_periodic(self):
        """
        2d array test, 1 ghost, periodic boundary conditions
        """

        # create the dist array, the sizes are local to each processor
        da = pnumpy.gdaZeros( (2,3), numpy.float32, numGhosts=1 )

        # processor rank and number of processes
        rk = da.rk
        nprocs = da.sz

        # set the data
        da[:] = rk

        # access neighbor data, collective operation
        southData = da.getData( (rk-1) % nprocs, winID=(1,0) )

        # check 
        self.assertEqual(southData.min(), (rk - 1) % nprocs)
        self.assertEqual(southData.max(), (rk - 1) % nprocs)

        # clean up
        da.free()

    def test2d_1_non_periodic(self):
        """
        2d array test, 1 ghost, non-periodic boundary conditions
        """

        # create the dist array, the sizes are local to each processor
        da = pnumpy.gdaZeros( (2,3), numpy.float32, numGhosts=1 )

        # processor rank and number of processes
        rk = da.rk
        nprocs = da.sz

        # set the data
        da[:] = rk

        # get the neighbor MPI rank (None if there is no neighbor)
        otherRk = rk - 1
        if otherRk < 0:
            otherRk = None

        # collective operation. all procs must call "get"
        southData = da.getData( otherRk, winID=(1,0) )

        # check 
        if otherRk is not None and otherRk >= 0:
            self.assertEqual(southData.min(), rk - 1)
            self.assertEqual(southData.max(), rk - 1)

        # clean up
        da.free()

    def test2d_laplacian_periodic(self):
        """
        2d array, apply Laplacian, periodic along the two axes
        """
        from pnumpy import CubeDecomp
        from pnumpy import MultiArrayIter
        import operator
        from math import sin, pi

        # global sizes
        ndims = 2
        #ns = numpy.array([60] * ndims)
        ns = numpy.array([3*4] * ndims)

        # local rank and number of procs
        rk = MPI.COMM_WORLD.Get_rank()
        sz = MPI.COMM_WORLD.Get_size()

        # find a domain decomposition
        dc = CubeDecomp(sz, ns)

        # not all numbers of procs will give a uniform domain decomposition,
        # exit if none can be found
        if not dc.getDecomp():
            if rk == 0: 
                print('no decomp could be found, adjust the number of procs')
            return
        
        # get the local start/stop indices along each axis as a list of 
        # 1d slices
        localSlices = dc.getSlab(rk)
        iBeg = numpy.array([s.start for s in localSlices])
        iEnd = numpy.array([s.stop for s in localSlices])
        nsLocal = numpy.array([s.stop - s.start for s in localSlices])
        
        # create the dist arrays
        da = pnumpy.gdaZeros(nsLocal, numpy.float32, numGhosts=1)
        laplacian = pnumpy.gdaZeros(nsLocal, numpy.float32, numGhosts=1)

        # set the data
        for it in MultiArrayIter(nsLocal):
            localInds = it.getIndices()
            globalInds = iBeg + localInds
            # positions are cell centered, domain is [0, 1]^ndims
            position = (globalInds  + 0.5)/ numpy.array(ns, numpy.float32)
            # sin(2*pi*x) * sin(2*pi*y) ...
            da[tuple(localInds)] = reduce(operator.mul, 
                                   [numpy.sin(2*numpy.pi*position[i]) for i in range(ndims)])

        # apply the Laplacian finite difference operator.
        # Start by performing all the operations that do
        # not require any communication.
        laplacian[:] = 2 * ndims * da

        # now subtract the neighbor values which are local to this process
        for idim in range(ndims):
            # indices shifted in the + direction along axis idim
            slabP = [slice(None, None, None) for j in range(idim)] + \
                [slice(1, None, None)] + \
                [slice(None, None, None) for j in range(idim + 1, ndims)]
            # indices shifted in the - direction along axis idim
            slabM = [slice(None, None, None) for j in range(idim)] + \
                [slice(0, -1, None)] + \
                [slice(None, None, None) for j in range(idim + 1, ndims)]
            laplacian[slabP] -= da[slabM] # subtract left neighbor
            laplacian[slabM] -= da[slabP] # subtract right neighbor

        # fetch the data located on other procs
        periodic = [True for idim in range(ndims)]
        for idim in range(ndims):
            # define the positive and negative directions
            directionP = tuple([0 for j in range(idim)] + [1] + [0 for j in range(idim + 1, ndims)])
            directionM = tuple([0 for j in range(idim)] + [-1] + [0 for j in range(idim + 1, ndims)])
            procP = dc.getNeighborProc(rk, directionP, periodic=periodic)
            procM = dc.getNeighborProc(rk, directionM, periodic=periodic)

            # this is where communication takes place... Note that when
            # accessing the data on the low-end side on rank procM we
            # access the slide on the positive side on procM (directionP).
            # And inversely for the high-end side data...
            dataM = da.getData(procM, winID=directionP)
            dataP = da.getData(procP, winID=directionM)

            # finish off the operator
            laplacian[da.getEllipsis(winID=directionM)] -= dataM
            laplacian[da.getEllipsis(winID=directionP)] -= dataP

        # compute a checksum and send the result to rank 0
        checksum = laplacian.reduce(lambda x,y:abs(x) + abs(y), 0.0, rootPe=0)
        if rk == 0:
            print('checksum = ', checksum)
            # float32 calculation has higher error
            assert(abs(checksum - 32.0) < 1.e-4)
        
        # free the windows
        da.free()
        laplacian.free()
       
if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPNumpy)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
    
