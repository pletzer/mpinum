#!/usr/bin/env python

# external dependencies

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
        self.index = 0

        # determine the locations of the non-zero displacement values
        nonZeroLocs = []
        for i in range(self.ndims):
            if disp[i] != 0:
               nonZeroLocs.append(i) 

        # list of unit displacements, ie all the displacements along the 
        # non-zero values of disp and their cross directions. For instance
        # if disp = (1, 0, -1) then dispUnits = [(0, 0, 0), (1, 0, 0),
        # (0, 0, -1), (1, 0, -1)]
        dispUnits = []
        for it in MultiArrayIter([2] * len(nonZeroLocs)):
            # inds are the indices that are active (1)/passive (0)
            inds = it.getIndices()
            du = [0] * self.ndims
            for i in range(len(nonZeroLocs)):
                k = nonZeroLocs[i]
                du[k] = inds[i] * disp[k]
            dispUnits.append(du)
        print('dispUnits = {}'.format(dispUnits))

        # the source/destination ellipses and remote ranks
        self.srcDom = []
        self.dstDom = []
        self.remoteRk = []
        for du in dispUnits:

            # the negative of du
            nu = [-d for d in du]

            sDom = Partition(self.ndims)
            dDom = Partition(self.ndims)
            rk = None
            numNonZeros = 0
            for i in range(self.ndims):
                # skip if disp is zero in this direction
                if disp[i] == 0:
                    continue

                numNonZeros += 1
                if du[i] == 0:
                    # shift
                    sDom = sDom.shift(du)
                    dDom = dDom.shift(nu)
                else:
                    # extract
                    sDom = sDom.extract(du)
                    dDom = dDom.extract(nu)
            if numNonZeros > 0:
                rk = decomp.getNeighborProc(myRank, du, periodic=periodic)
            self.remoteRk.append(rk)
            self.srcDom.append(sDom)
            self.dstDom.append(dDom)

        self.ntot = len(dispUnits)

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
    for disp in (1, 1),: #(0, 0), (0, 1), (1, 1):
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
