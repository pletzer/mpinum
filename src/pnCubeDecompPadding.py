#!/usr/bin/env python

"""
Muti-dimensional domain decomposition with padding
"""

# standard modules
from functools import reduce
import operator
import math
import numpy
from scipy.optimize import minimize

# pnumpy modules
from pnumpy.pnMultiArrayIter import MultiArrayIter


class CubeDecompPadding:

    def __init__(self, nprocs, dims, rowMajor=True, fill=float('nan')):
        """
        Constructor
        @param nprocs number of sub-domains
        @param dims list of global dimensions along each axis
        @param rowMajor True if row major, False if column major
               (determines whether the processor ranks are contiguous
                as the first or last index increases)
        @param fill fill value for the data that need padding
        """
        self.ndims = len(dims)
        self.nprocs = nprocs
        self.globalDims = dims
        self.rowMajor = rowMajor

        # total number of degrees of freedom (without padding)
        self.ntot = reduce(operator.mul, self.globalDims, 1)

        # list of number of procs per axis
        self.decomp = None

        # list of local sizes per axis
        self.localDims = []

        # maps the processor rank to start/end index sets
        self.proc2IndexSet = {}

        # iterator from one sub-domain to the next
        self.mit = None

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

    def getSlab(self, procId):
        """
        Get the start/end indices for given processor rank
        @param procId processor rank
        @return list of slices (or empty list if no valid decomp)
        """
        if self.proc2IndexSet:
            return self.proc2IndexSet[procId]
        else:
            return []

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
        same volume in index space. Padding is applied to ensure 
        a uniform domain decomposition
        """
        lmbda0 = 1.0
        lmbda1 = 0.01

        def xCeil(x):
            return numpy.array([int(math.ceil(s)) for s in x])

        def surface(x):
            res = 1
            for i in range(self.ndims):
                sz = [s for s in x]
                sz[i] = 1
                res += 2*reduce(operator.mul, sz, 1)
            return res

        def volume(x):
            return reduce(operator.mul, x, 1)

        def matchNumberProcs(x):
            xC = xCeil(x)
            nprocs = reduce(operator.mul, [self.globalDims[i]//xC[i] for i in range(self.ndims)], 1)
            return nprocs - self.nprocs

        def costFunction(x):
            dxC = xCeil(x) - x
            surf = surface(x)
            res = lmbda0*surf/volume(x) + lmbda1*numpy.dot(dxC, dxC)/surf
            return res

        nPerAxis = self.nprocs**(1./self.ndims)
        x0 = [d/float(nPerAxis) for d in self.globalDims]
        cons = ({'type': 'eq', 
                 'fun': matchNumberProcs})
        opts = {'ftol': 1.e-6,
                'eps': 1.e-3,
                'disp': True,
                'maxiter': 100}
        opt = minimize(costFunction, x0, method='SLSQP', options=opts, constraints=cons)
        if not opt.success:
            print('*** ERROR failed to find minimum, number of iterations: {}'.format(opt.nit
                ))
        # need to check the solution
        # get the solution
        self.localDims = xCeil(opt.x)
        print('*** self.localDims = {}'.format(self.localDims))
        self.decomp = [self.globalDims[i]//self.localDims[i] for i in range(self.ndims)]


######################################################################


def test():

    dims = (46, 72)
    offsetN = (1, 0)
    offsetE = (0, 1)
    offsetS = (-1, 0)
    offsetW = (0, -1)

    for nprocs in 1, 2, : # 3,: # 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18:
        d = CubeDecompPadding(nprocs=nprocs, dims=dims)
        print('nprocs = {0} decomp: {1}'.format(nprocs, d.getDecomp()))
        """
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
        """

if __name__ == '__main__':
    test()
