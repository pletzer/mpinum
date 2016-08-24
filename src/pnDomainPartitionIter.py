#!/usr/bin/env python

# external dependencies
import numpy

# internal dependencies
from pnumpy import Partition
from pnumpy import MultiArrayIter
from pnumpy import CubeDecomp

class DomainPartitionIter:

    def __init__(self, disp, decomp, myRank, periodic=None):
        """
        Constructor
        @param disp displacement vector
        @param decomp decomposition 
        @param myRank my rank
        @param periodic list of True/False values (True = periodic)
        """
        self.disp = disp
        self.ndims = len(disp)
        self.index = -1

        # determine the locations of the non-zero displacement values
        nonZeroLocs = []
        for i in range(self.ndims):
            if disp[i] != 0:
               nonZeroLocs.append(i) 

        # list of unit displacements, all elements are 0 except one
        dispUnits = []
        for loc in nonZeroLocs:
            du = [0] * self.ndims
            du[loc] = disp[loc]
            dispUnits.append(numpy.array(du))
        #print(dispUnits)

        self.srcDom = []
        self.dstDom = []
        self.remoteRk = []
        for it in MultiArrayIter([2] * len(nonZeroLocs)):
            # inds refer to non-zero indices of disp that require extract (1)/shift (0) operation
            inds = it.getIndices()
            sDom = Partition(self.ndims)
            dDom = Partition(self.ndims)
            dirct = [0] * self.ndims
            for i in range(len(inds)):
                du = dispUnits[i]
                loc = nonZeroLocs[i]
                if inds[i] == 0:
                    sDom = sDom.shift(du)
                    dDom = dDom.shift(-du)
                else:
                    dirct[loc] = -disp[loc]
                    sDom = sDom.extract(du)
                    dDom = dDom.extract(-du)
            self.srcDom.append(sDom)
            self.dstDom.append(dDom)
            rk = decomp.getNeighborProc(myRank, dirct, periodic=periodic)
            self.remoteRk.append(rk)
            print('...inds = {0} sDom = {1} dDom = {2} remote rank = {3}'.format(inds, sDom.getSlice(), dDom.getSlice(), rk))

        self.ntot = len(self.srcDom)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.ntot - 1:
            self.index += 1
            return self
        else:
            raise StopIteration

    # Python2
    def next(self):
        return self.__next__()

    def getSrcPartition(self):
        return self.srcDom[self.index]

    def getDstPartition(self):
        """
        """
        return self.dstDom[self.index]

    def getRemoteRank(self):
        return self.remoteRk[self.index]

    def getWindowId(self):
        return self.getStringFromPartition(self.srcDom[self.index].getSlice())

    def getStringFromPartition(self, partition):
        """
        Get the string representation of the partition
        @param partition a sequence of python slice objects
        @return string like ":-1,0:2"
        """
        res = ''
        for s in partition:
            start = ''
            stop = ''
            if s.start is not None:
                start = int(s.start)
            if s.stop is not None:
                stop = int(s.stop)
            res += '{0}:{1},'.format(start, stop)
        return res


######################################################################################################
def test0d():
    nprocs = 1
    decomp = CubeDecomp(nprocs, dims=[])
    dmi = DomainPartitionIter(disp=(), decomp=decomp, myRank=0, periodic=[])
    for d in dmi:
        print('test0d: getSrcPartition => {}'.format(d.getSrcPartition()))

def test1d():
    nprocs = 1
    decomp = CubeDecomp(nprocs, dims=[2])
    dmi = DomainPartitionIter(disp=(1,), decomp=decomp, myRank=0, periodic=[True])
    for d in dmi:
        print('test1d: getSrcPartition => {}'.format(d.getSrcPartition()))

def test2d():
    nprocs = 1
    decomp = CubeDecomp(nprocs, dims=[2, 3])
    for disp in (0, 0), (0, 1): #(1, 1),: #(0, 0), (0, 1), (1, 1):
        print('='*40)
        print('test2d: disp = {}'.format(disp))
        dmi = DomainPartitionIter(disp=disp, decomp=decomp, myRank=0, periodic=[True, False])
        for d in dmi:
            print('    src partition {}'.format(d.getSrcPartition()))
            print('    dst partition {}'.format(d.getDstPartition()))
            print('    remote rank   {}'.format(d.getRemoteRank()))
            print('    window Id     {}'.format(d.getWindowId()))
            print('-'*20)

if __name__ == '__main__':
    #test0d()
    #test1d()
    test2d()
