#!/usr/bin/env python

"""
Apply Laplacian stencil to distributed array data
"""

# external dependencies
from mpi4py import MPI
import numpy
from mpinum import gdaZeros, Partition


class Laplacian:

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

        # zero displacement vector
        self.zeros = tuple([0] * self.ndims)

        # set the laplacian stencil weights
        self.stencil = {
            self.zeros: -2.0 * self.ndims,
            }
        for drect in range(self.ndims):
            for pm in (-1, 1):
                disp = [0] * self.ndims
                disp[drect] = pm
                self.stencil[tuple(disp)] = 1.0

        #
        # build the domain partitioning/topology data structures, all
        # of these take the displacement vector as input
        #
        # entire domain
        self.domain = Partition(self.ndims)

        # the local domain of the input array
        self.srcLocalDomains = {}
        # the local domain of the output array
        self.dstLocalDomains = {}

        # the side domain on the neighbor rank
        self.srcSlab = {}
        # the side domain on the receiving end
        self.dstSlab = {}

        # the window Ids
        self.winIds = {}

        # the neighbor rank
        self.neighRk = {}

        for drect in range(self.ndims):
            for pm in (-1, 1):
                disp = [0] * self.ndims
                disp[drect] = pm
                disp = tuple(disp)
                # negative displacements
                nisp = [0] * self.ndims
                nisp[drect] = -pm
                nisp = tuple(nisp)
                self.srcLocalDomains[disp] = self.domain.shift(disp).getSlice()
                self.dstLocalDomains[disp] = self.domain.shift(nisp).getSlice()
                self.srcSlab[disp] = self.domain.extract(disp).getSlice()
                self.dstSlab[disp] = self.domain.extract(nisp).getSlice()

                # assumes disp only contains -1, 0s, or 1
                self.neighRk[disp] = decomp.getNeighborProc(myRank,
                                                            disp,
                                                            periodic=periodic)

                self.winIds[disp] = nisp

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

        # no displacement term
        weight = self.stencil[self.zeros]
        out[...] += weight * localArray

        for disp in self.srcLocalDomains:

            weight = self.stencil[disp]

            # no communication required here
            srcDom = self.srcLocalDomains[disp]
            dstDom = self.dstLocalDomains[disp]

            out[dstDom] += weight * localArray[srcDom]

            #
            # now the part that requires communication
            #

            # set the ghost values
            srcSlab = self.srcSlab[disp]
            # copy
            inp[srcSlab] = localArray[srcSlab]

            # send over to local process
            dstSlab = self.dstSlab[disp]
            winId = self.winIds[disp]
            rk = self.neighRk[disp]

            # remote fetch
            out[dstSlab] += weight * inp.getData(rk, winId)

        # some implementations require this
        inp.free()

        return out
