#!/usr/bin/env python

"""
Apply stencil to distributed array data
"""

# external dependencies
from mpi4py import MPI
import numpy

# internal dependencies
from pnumpy import gdaZeros
from pnumpy import Partition
from pnumpy import MultiArrayIter
from pnumpy import DomainPartitionIter


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
        self.myRank = comm.Get_rank()
        self.comm = comm

        # defaul stencil is empty
        self.stencil = {}

        # partition logic, initially empty
        self.dpis = {}

    def addStencilBranch(self, disp, weight):
        """
        Set or overwrite the stencil weight for the given direction
        @param disp displacement vector
        @param weight stencil weight
        """
        self.stencil[disp] = weight
        self.__setPartionLogic(disp)

    def removeStencilBranch(self, disp):
        """
        Remove a stencil branch
        @param disp displacement vector
        """
        del self.stencil[disp]
        del self.dpsi[disp]

    def __setPartionLogic(self, disp):

        sdisp = str(disp)

        srcDp = DomainPartitionIter(disp)
        dstDp = DomainPartitionIter([-d for d in disp])

        srcs = [d.getSlice() for d in srcDp]
        dsts = [d.getSlice() for d in dstDp]

        srcDp.reset()
        remoteRanks = [decomp.getRemoteRank(self.myRank, part.getDirection(), 
                                                periodic=self.periodic) \
                       for part in srcDp]
            
        srcDp.reset()
        remoteWinIds = [sdisp + '[' + part.getStringPartition() + ']' \
                        for part in srcDp]

        self.dpis[disp] = {
            'srcs': srcs,
            'dsts': dsts,
            'remoteRanks': remoteRanks,
            'remoteWinIds': remoteWinIds,
        }


    def apply(self, localArray):
        """
        Apply stencil to data
        @param localArray local array
        @return new array on local proc
        """

        # input dist array
        inp = daZeros(localArray.shape, localArray.dtype)
        inp.setComm(self.comm)

        # output array
        out = numpy.zeros(localArray.shape, localArray.dtype)

        # expose the dist array windows
        for disp, dpi in self.dpis.items():

            sdisp = str(disp)

            srcs = dpi['srcs']
            remoteWinIds = dpi['remoteWinIds']
            numParts = len(srcs)
            for i in range(numParts):
                inp.expose(srcs[i], winID=remoteWinIds[i])

        # apply the stencil
        for disp, weight in self.stencil.items():

            dpi = self.dpis[disp]

            dpi = self.dpis[disp]

            srcs = dpi['srcs']
            dsts = dpi['dsts']
            remoteRanks = dpi['remoteRanks']
            remoteWinIds = dpi['remoteWinIds']
            numParts = len(srcs)
            for i in range(numParts):
                srcSlce = srcs[i]
                dstSlce = dsts[i]
                remoteRank = remoteRanks[i]
                remoteWinId = remoteWinIds[i]

                # now apply the stencil
                out[dstSlce] += weight * inp.getData(remoteRank, remoteWinId)

        # some implementations require this
        inp.free()

        return out

##############################################################################


def test1d():
    so = StencilOperator(decomp, periodic=[True])


if __name__ == '__main__':
    test1d()