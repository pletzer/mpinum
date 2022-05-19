#!/usr/bin/env python

# external dependencies
import numpy

# internal dependencies
from mpinum import Partition
from mpinum import MultiArrayIter


class DomainPartitionIter:

    def __init__(self, disp):
        """
        Constructor
        @param disp displacement vector
        """
        self.disp = disp
        self.ndims = len(disp)

        # counter
        self.index = -1

        # index locations where disp is non-zero
        nonZeroLocs = []
        for i in range(self.ndims):
            if disp[i] != 0:
                nonZeroLocs.append(i)

        # list of unit displacements, all combinations of
        # vectors pointing in the diection of dispo. Eg
        # if disp =  (1, 0, -1) then dispUnits should be
        # [(1, 0, 0), (0, 0, -1)]
        dispUnits = []
        for loc in nonZeroLocs:
            du = [0] * self.ndims
            du[loc] = disp[loc]
            dispUnits.append(numpy.array(du))

        # list of partitions
        self.partitions = []

        # list of directions
        self.directions = []

        # for each dispUnits apply either shift or extract operation.
        # Number of partitions is 2**len(dispUnits)
        for it in MultiArrayIter([2] * len(dispUnits)):

            # elements of inds:
            # 0 => shift
            # 1 => extract
            inds = it.getIndices()

            # create entire domain partition
            part = Partition(self.ndims)

            drctn = numpy.zeros(self.ndims, numpy.int32)

            # iterate over the unit displacements
            for i in range(len(inds)):

                # unit displacement
                du = dispUnits[i]

                # axis index for this unit displacement
                loc = nonZeroLocs[i]

                # decide if it will be a shift or an extract
                if inds[i] == 0:
                    part = part.shift(du)
                else:
                    part = part.extract(du)
                    drctn += numpy.maximum(-1, numpy.minimum(1, du))

            self.partitions.append(part)
            self.directions.append(drctn)

        # number of partitions
        self.numParts = len(self.partitions)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.numParts - 1:
            self.index += 1
            return self
        else:
            raise StopIteration

    # Python2
    def next(self):
        return self.__next__()

    def getNumberOfPartitions(self):
        """
        Get the number of partitions
        @return number
        """
        return self.numParts

    def reset(self):
        """
        Reset counter
        """
        self.index = -1

    def getDirection(self):
        """
        Get the current direction pointing to the neighbor rank
        """
        return self.directions[self.index]

    def getPartition(self):
        """
        Get the current partition
        @return object
        """
        return self.partitions[self.index]

    def getStringPartition(self):
        """
        Get the string representation of the current partition
        @return string like ":-1,0:2"
        """
        res = ''
        for s in self.partitions[self.index].getSlice():
            start = ''
            stop = ''
            if s.start is not None:
                start = int(s.start)
            if s.stop is not None:
                stop = int(s.stop)
            res += '{0}:{1},'.format(start, stop)
        return res


##############################################################################
def test0d():
    print('='*40)
    for disp in (), :
        print('test0d: disp = {}'.format(disp))
        dmi = DomainPartitionIter(disp=disp)
        for d in dmi:
            print('    partition {}'.format(d.getStringPartition()))


def test1d():
    print('='*40)
    for disp in (1,), (-1,):
        print('test1d: disp = {}'.format(disp))
        dmi = DomainPartitionIter(disp=disp)
        for d in dmi:
            print('    partition {}'.format(d.getStringPartition()))


def test2d():
    print('='*40)
    for disp in (0, 0), (0, 1), (1, 1):
        print('test2d: disp = {}'.format(disp))
        dmi = DomainPartitionIter(disp=disp)
        for d in dmi:
            print('    partition {}'.format(d.getStringPartition()))


def test3d():
    print('='*40)
    for disp in (0, 0, 0), (0, 0, 1), (1, 0, -1), (-1, -1, -1):
        print('test3d: disp = {}'.format(disp))
        dmi = DomainPartitionIter(disp=disp)
        for d in dmi:
            print('    partition {}'.format(d.getStringPartition()))

if __name__ == '__main__':
    test0d()
    test1d()
    test2d()
    test3d()
