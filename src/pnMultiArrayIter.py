#!/usr/bin/env python

"""
Multi-array iterator class.
"""

# standard modules
import operator
from functools import reduce
import numpy


class MultiArrayIter:

    def __init__(self, dims, rowMajor=True):
        """
        Constructor
        @param dims list of dimensions along each axis
        @param rowMajor True if row major, False if column major
        """
        self.dims = dims
        self.ntot = reduce(operator.mul, self.dims, 1)
        self.ndims = len(self.dims)
        self.big_index = -1
        self.dimProd = numpy.array([1 for i in range(self.ndims)])
        if rowMajor:
            # row major
            for i in range(self.ndims - 2, -1, -1):
                self.dimProd[i] = self.dimProd[i + 1] * self.dims[i + 1]
        else:
            # column major
            for i in range(1, self.ndims):
                self.dimProd[i] = self.dimProd[i - 1] * self.dims[i - 1]

    def __iter__(self):
        return self

    def __next__(self):
        if self.big_index < self.ntot - 1:
            self.big_index += 1
            return self
        else:
            raise StopIteration

    # Python2
    def next(self):
        return self.__next__()

    def getIndices(self):
        """
        @return current index set
        """
        return self.getIndicesFromBigIndex(self.big_index)

    def getBigIndex(self):
        """
        @return current big index
        """
        return self.big_index

    def getIndicesFromBigIndex(self, bigIndex):
        """
        Get index set from given big index
        @param bigIndex
        @return index set
        @note no checks are performed to ensure that the returned
        big index is valid
        """
        indices = numpy.array([0 for i in range(self.ndims)])
        for i in range(self.ndims):
            indices[i] = bigIndex // self.dimProd[i] % self.dims[i]
        return indices

    def getBigIndexFromIndices(self, indices):
        """
        Get the big index from a given set of indices
        @param indices
        @return big index
        @note no checks are performed to ensure that the returned
        indices are valid
        """
        return reduce(operator.add, [self.dimProd[i]*indices[i]
                                     for i in range(self.ndims)], 0)

    def reset(self):
        """
        Reset big index
        """
        self.big_index = -1

    def getDims(self):
        """
        Get the axis dimensions
        @return list
        """
        return self.dims

    def isBigIndexValid(self, bigIndex):
        """
        Test if big index is valid
        @param bigIndex big index
        @return True if big index is in range, False otherwise
        """
        return bigIndex < self.ntot and bigIndex >= 0

    def areIndicesValid(self, inds):
        """
        Test if indices are valid
        @param inds index set
        @return True if valid, False otherwise
        """
        return reduce(operator.and_, [0 <= inds[d] < self.dims[d]
                                      for d in range(self.ndims)], True)

    def getNumberOfElements(self):
        """
        Get the number of elements
        @return number
        """
        return self.ntot


######################################################################


def test(rowMajor):
    dims = (2, 3, 4)
    print('row major: dims = {0}'.format(dims))
    for it in MultiArrayIter((2, 3, 4), rowMajor=rowMajor):
        inds = it.getIndices()
        bi = it.getBigIndex()
        print('indices = {0} big index = {1}'.format(inds, bi))
        assert(it.getBigIndexFromIndices(inds) == it.getBigIndex())
        inds2 = it.getIndicesFromBigIndex(bi)
        assert(reduce(operator.and_, [inds2[d] == inds[d]
                                      for d in range(it.ndims)], True))
        assert(it.isBigIndexValid(bi))
        assert(it.areIndicesValid(inds))

    # check validity
    assert(not it.isBigIndexValid(-1))
    assert(not it.isBigIndexValid(it.ntot))
    assert(not it.areIndicesValid((2, 2, 3)))
    assert(not it.areIndicesValid((1, 3, 3)))
    assert(not it.areIndicesValid((1, 2, 4)))

if __name__ == '__main__':
    test(rowMajor=True)
    test(rowMajor=False)
