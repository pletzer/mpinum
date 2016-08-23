#!/usr/bin/env python

"""
Partition of a domain in index space
"""

# internal dependencies
import copy

# external dependencies


class Partition:

    def __init__(self, ndims):
        """
        Constructor
        @param ndims number of dimensions
        """
        self.ndims = ndims
        # the entire domain
        self.domain = [slice(0, None) for i in range(ndims)]

    def getSlice(self):
        """
        Return a slice data structure that can be used as indexing
        in numpy arrays
        @return tuple of slice objects
        """
        return tuple(self.domain)

    def shift(self, disp):
        """
        Shift operation
        @param displacement vector in index space
        @return domain shifted to the right/left
        """
        res = copy.deepcopy(self)
        for i in range(self.ndims):
            d = disp[i]
            s = self.domain[i]
            if d > 0:
                res.domain[i] = slice(s.start + d, s.stop)
            elif d < 0:
                res.domain[i] = slice(s.start, int(s.stop or 0) + d)
        return res

    def extract(self, disp):
        """
        Extraction operation
        @param displacement vector in index space
        @return the part of the domain that is exposed by the shift
        """
        res = copy.deepcopy(self)
        for i in range(self.ndims):
            d = disp[i]
            s = self.domain[i]
            if d > 0:
                res.domain[i] = slice(s.start, d)
            elif d < 0:
                res.domain[i] = slice(d, s.stop)
        return res

    def __str__(self):
        res = "num dims: {}\n".format(self.ndims)
        count = 0
        for s in self.domain:
            res += "dim: {0} index range [{1}, {2}(\n".format(count,
                                                              s.start,
                                                              s.stop)
            count += 1
        return res

##############################################################################


def test0d():
    p = Partition(0)
    print(p.shift(()))


def test1d():
    p = Partition(1)
    print(p.shift((1,)))
    print(p.shift((-1,)))


def test2d():
    p = Partition(2)
    print(p.shift((0, -1)))
    print(p.extract((0, -1)))


def test3d():
    p = Partition(3)
    print(p.shift((1, -2, 3)))

def testCross2d():
    p = Partition(2)
    print('p.shift((1, -1))')
    print(p.shift((1, -1)))
    print('p.shift((1, 0)).shift((0, -1))')
    print(p.shift((1, 0)).shift((0, -1)))
    print('p.extract((1, -1))')
    print(p.extract((1, -1)))

if __name__ == '__main__':
    test0d()
    test1d()
    test2d()
    test3d()
    testCross2d()
