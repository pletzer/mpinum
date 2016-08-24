#!/usr/bin/env python

"""
Apply stencil to distributed array data
"""

# external dependencies
from mpi4py import MPI
import numpy

# internal dependencies
from pnumpy import gdaZeros, Partition, MultiArrayIter


class StencilOperator:

    def __init__(self, decomp, periodic=None, comm=MPI.COMM_WORLD):
        """
        Constructor
        @param decomp instance of setCubeDecomp
        @param periodic list of True/False values (True for periodic)
        @param comm MPI communicator
        """
        # number of dimensions
        self.ndims = decomp.getNumDims()

        # this process's MPI rank
        myRank = comm.Get_rank()

        # stencil, by default Laplacian, keep zero weight
        # branches
        for it in MultiArrayIter([3] * self.ndims):
            disp = tuple(e - 1 for e in it.getIndices()])
            self.stencil[disp] = 0.0
            numNonZero = self.getNumNonZero(disp) 
            if numNonZero == 1:
                self.stencil[disp] = 1.0
            elif numNonZero == 0:
                self.stencil[disp] = -4.0

        self.dpis = {}
        for disp in self.stencil:
            self.dpsi[disp] = DomainPartitionIter(disp, decomp)

    def setStencilWeight(self, disp):
        """
        Set or overwrite the stencil weight for the given direction
        @param disp displacement vector
        @param weight stencil weight
        """
        self.stencil[disp] = weight

    def removeStencilBranch(self, disp):
        """
        Remove a stencil branch
        @param disp displacement vector
        """
        del self.stencil[disp]
        del self.dpsi[disp]

    def apply(self, localArray):
        """
        Apply Laplacian stencil to data
        @param localArray local array
        @return new array on local proc
        """

        # input dist array
        inp = gdaZeros(localArray.shape, localArray.dtype, numGhosts=1)
        # output array
        out = numpy.zeros(localArray.shape, localArray.dtype)

        for disp, weight in self.stencil.items():
            dpi = self.dpis[disp]
            for it in dpi:
                srcPart = it.getSrcPartition()
                dstPart = it.getDstPartition()
                remoteRk = it.getRemoteRank()
                if remoteRk is None:
                    # local, no communication
                    out[dstPart] += weight * inp[srcPart]
                else:
                    winId = it.getWindowId()
                    if winId is not None:
                        out[dstPart] += weight * inp.getData(remoteRk, winID)

        # some implementations require this
        inp.free()

        return out
