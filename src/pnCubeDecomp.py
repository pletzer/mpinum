#!/usr/bin/env python

"""
Muti-dimensional domain decomposition.
"""

# standard modules
import operator
from functools import reduce
from mpinum.pnMultiArrayIter import MultiArrayIter


def getPrimeFactors(n):
    """
    Get all the prime factor of given integer
    @param n integer
    @return list [1, ..., n]
    """
    lo = [1]
    n2 = n // 2
    k = 2
    for k in range(2, n2 + 1):
        if (n // k)*k == n:
            lo.append(k)
    return lo + [n, ]


class CubeDecomp:

    def __init__(self, nprocs, dims, rowMajor=True):
        """
        Constructor
        @param nprocs number of sub-domains
        @param dims list of global dimensions along each axis
        @param rowMajor True if row major, False if column major
               (determines whether the processor ranks are contiguous
                as the first or last index increases)
        """
        self.ndims = len(dims)
        self.nprocs = nprocs
        self.globalDims = dims
        self.rowMajor = rowMajor

        # holds a list of number of procs per axis
        self.decomp = None

        # maps the processor rank to start/end index sets
        self.proc2IndexSet = {}

        # iterator from one sub-domain to the next
        self.mit = None

        # list of valid number of procs
        self.validProcs = []

        self.__computeDecomp()

    def getNumDims(self):
        """
        Get the number of topological dimensions
        """
        return self.ndims

    def getDecomp(self):
        """
        Get the decomposition
        @return list of number of procs per axis, or None if
        no decomp exists for this number of processors
        """
        return self.decomp

    def isValid(self):
        """
        Is decomposition valid? 
        @return True/False
        """
        if self.decomp:
            return True
        else:
            return False

    def getSlab(self, procId):
        """
        Get the start/end indices for given processor rank
        @param procId processor rank
        @return tuple of slices (or empty tuple if no valid decomp)
        """
        if self.proc2IndexSet:
            return tuple(self.proc2IndexSet[procId])
        else:
            return ()

    def getNeighborProc(self, proc, offset, periodic=None):
        """
        Get the neighbor to a processor
        @param proc the reference processor rank
        @param offset displacement, e.g. (1, 0) for north, (0, -1) for west,...
        @param periodic boolean list of True/False values, True if axis is
                        periodic, False otherwise
        @note will return None if there is no neighbor
        """

        if self.mit is None:
            # no decomp, just exit
            return None

        inds = [self.mit.getIndicesFromBigIndex(proc)[d] + offset[d]
                for d in range(self.ndims)]

        if periodic is not None and self.decomp is not None:
            # apply modulo operation on periodic axes
            for d in range(self.ndims):
                if periodic[d]:
                    inds[d] = inds[d] % self.decomp[d]

        if self.mit.areIndicesValid(inds):
            return self.mit.getBigIndexFromIndices(inds)
        else:
            return None

    def __computeDecomp(self):
        """
        Compute optimal dedomposition, each sub-domain has the
        same volume in index space.
        @return list if successful, empty list if not successful
        """
        primeNumbers = [getPrimeFactors(d) for d in self.globalDims]

        ns = [len(pns) for pns in primeNumbers]
        validDecomps = []
        self.validProcs = []
        for it in MultiArrayIter(ns):
            inds = it.getIndices()
            decomp = [primeNumbers[d][inds[d]] for d in range(self.ndims)]
            self.validProcs.append(reduce(operator.mul, decomp, 1))
            if reduce(operator.mul, decomp, 1) == self.nprocs:
                validDecomps.append(decomp)

        # sort and remove duplicates
        self.validProcs.sort()
        vprocs = []
        for vp in self.validProcs:
            if len(vprocs) == 0 or (len(vprocs) >= 1 and vp != vprocs[-1]):
                vprocs.append(vp)
        self.validProcs = vprocs

        if len(validDecomps) == 0:
            # no solution
            return

        # find the optimal decomp among all valid decomps
        minCost = float('inf')
        bestDecomp = validDecomps[0]
        for decomp in validDecomps:
            sizes = [self.globalDims[d]//decomp[d] for d in range(self.ndims)]
            volume = reduce(operator.mul, sizes, 1)
            surface = 0
            for d in range(self.ndims):
                surface += 2*reduce(operator.mul, sizes[:d], 1) * \
                    reduce(operator.mul, sizes[d+1:], 1)
            cost = surface / float(volume)
            if cost < minCost:
                bestDecomp = decomp
                minCost = cost
        self.decomp = bestDecomp

        # ok, we have a valid decomp, now build the sub-domain iterator
        self.mit = MultiArrayIter(self.decomp, rowMajor=self.rowMajor)

        # fill in the proc to index set map
        procId = 0
        self.proc2IndexSet = {}
        numCellsPerProc = [self.globalDims[d]//self.decomp[d]
                           for d in range(self.ndims)]
        for it in self.mit:
            nps = it.getIndices()
            self.proc2IndexSet[procId] = []
            for d in range(self.ndims):
                sbeg = nps[d]*numCellsPerProc[d]
                send = (nps[d] + 1)*numCellsPerProc[d]
                self.proc2IndexSet[procId].append(slice(sbeg, send))
            procId += 1

    def getNumberOfValidProcs(self):
        """
        Get the number of valid processor counts
        @return number
        """
        return self.validProcs

######################################################################


def test():

    dims = (46, 72)
    offsetN = (1, 0)
    offsetE = (0, 1)
    offsetS = (-1, 0)
    offsetW = (0, -1)

    for nprocs in 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18:
        d = CubeDecomp(nprocs=nprocs, dims=dims)
        if nprocs == 1:
            print('valid prcessor counts: {0}'.format(
                d.getNumberOfValidProcs()))
        print('nprocs = {0} decomp: {1}'.format(nprocs, d.getDecomp()))
        for procId in range(nprocs):
            print('[{0}] start/end indices {1}: '.format(
                       procId, d.getSlab(procId)))
            procN = d.getNeighborProc(procId, offsetN)
            procE = d.getNeighborProc(procId, offsetE)
            procS = d.getNeighborProc(procId, offsetS)
            procW = d.getNeighborProc(procId, offsetW)
            print('      {0}     '.format(procN))
            print('{0}         {0}'.format(procW, procE))
            print('      {0}     '.format(procS))

if __name__ == '__main__':
    test()
