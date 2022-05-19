import mpinum
import numpy
import operator
import unittest
from mpi4py import MPI
from mpinum.pnMultiArrayIter import MultiArrayIter
from functools import reduce

class TestMultiArrayIter(unittest.TestCase):
    """
    Test mpinum
    """

    def setUp(self):
        pass

    def testRowMajor(self):
        dims = (2, 3, 4)
        print('row major: dims = {0}'.format(dims))
        for it in MultiArrayIter( (2, 3, 4), rowMajor = True):
            inds = it.getIndices()
            bi = it.getBigIndex()
            print('indices = {0} big index = {1}'.format(inds, bi))
            assert( it.getBigIndexFromIndices(inds) == it.getBigIndex() )
            inds2 = it.getIndicesFromBigIndex(bi)
            assert( reduce(operator.and_, \
                           [inds2[d] == inds[d] for d in range(it.ndims)], True ) )
            assert( it.isBigIndexValid(bi) )
            assert( it.areIndicesValid(inds) )

    def testColumnMajor(self):
        dims = (2, 3, 4)
        print('column major: dims = {0}'.format(dims))
        for it in MultiArrayIter( (2, 3, 4), rowMajor = False):
            inds = it.getIndices()
            bi = it.getBigIndex()
            print('indices = {0} big index = {1}'.format(inds, bi))
            assert( it.getBigIndexFromIndices(inds) == it.getBigIndex() )
            inds2 = it.getIndicesFromBigIndex(bi)
            assert( reduce(operator.and_, \
                           [inds2[d] == inds[d] for d in range(it.ndims)], True ) )
            assert( it.isBigIndexValid(bi) )
            assert( it.areIndicesValid(inds) )

       
if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiArrayIter)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
    
