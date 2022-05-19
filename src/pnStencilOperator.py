#!/usr/bin/env python

"""
Apply stencil to distributed array data
"""

# external dependencies
from mpi4py import MPI
import numpy

# internal dependencies
from mpinum import daZeros
from mpinum import DomainPartitionIter
from mpinum import CubeDecomp


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
        self.decomp = decomp
        self.periodic = periodic

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
        self.stencil[tuple(disp)] = weight
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

        srcs = [d.getPartition().getSlice() for d in srcDp]
        dsts = [d.getPartition().getSlice() for d in dstDp]

        srcDp.reset()
        remoteRanks = [self.decomp.getNeighborProc(self.myRank,
                                                   part.getDirection(),
                                                   periodic=self.periodic)
                       for part in srcDp]

        srcDp.reset()
        remoteWinIds = [sdisp + '[' + part.getStringPartition() + ']'
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
        inp[...] = localArray
        inp.setComm(self.comm)

        # output array
        out = numpy.zeros(localArray.shape, localArray.dtype)

        # expose the dist array windows
        for disp, dpi in self.dpis.items():

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
                if remoteRank == self.myRank:
                    # local updates
                    out[dstSlce] += weight*inp[srcSlce]
                else:
                    # remote fetch
                    out[dstSlce] += weight*inp.getData(remoteRank, remoteWinId)

        # some implementations require this
        inp.free()

        return out

##############################################################################


def test1d(disp, dtyp):
    rk = MPI.COMM_WORLD.Get_rank()
    sz = MPI.COMM_WORLD.Get_size()
    dims = (3,)
    globalDims = (3*sz,)
    decomp = CubeDecomp(nprocs=sz, dims=globalDims)
    so = StencilOperator(decomp, periodic=[True])
    so.addStencilBranch(disp, 2)
    inputData = (rk + 1) * numpy.ones(dims, dtyp)
    outputData = so.apply(inputData)
    print('[{0}] inputData = {1}'.format(rk, inputData))
    print('[{0}] outputData = {1}'.format(rk, outputData))
    MPI.COMM_WORLD.Barrier()

def test2d(disp, dtyp):
    rk = MPI.COMM_WORLD.Get_rank()
    sz = MPI.COMM_WORLD.Get_size()
    dims = (3, 3)
    globalDims = (3*sz, sz)
    decomp = CubeDecomp(nprocs=sz, dims=globalDims)
    so = StencilOperator(decomp, periodic=[True, True])
    so.addStencilBranch(disp, 2)
    inputData = (rk + 1) * numpy.ones(dims, dtyp)
    outputData = so.apply(inputData)
    print('[{0}] inputData = {1}'.format(rk, inputData))
    print('[{0}] outputData = {1}'.format(rk, outputData))
    MPI.COMM_WORLD.Barrier()


if __name__ == '__main__':

    myRank = MPI.COMM_WORLD.Get_rank()

    if myRank == 0: print('== test1d int16 ==')
    disp = (1,)
    test1d(disp, numpy.int16)

    if myRank == 0: print('== test1d int32 ==')
    disp = (1,)
    test1d(disp, numpy.int32)

    if myRank == 0: print('== test1d int64 ==')
    disp = (-1,)
    test1d(disp, numpy.int64)

    if myRank == 0: print('== test1d float32 ==')
    disp = (-1,)
    test1d(disp, numpy.float32)

    if myRank == 0: print('== test1d float64 ==')
    disp = (-1,)
    test1d(disp, numpy.float64)

    if myRank == 0: print('== test2d float32 ==')
    disp = (1, 0)
    test2d(disp, numpy.float32)
