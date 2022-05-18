import mumpy
import numpy
import unittest
from functools import reduce
from mpi4py import MPI

class TestCubeDecomp(unittest.TestCase):
    """
    Test Cubedecomp
    """

    def setUp(self):
        self.rk = MPI.COMM_WORLD.Get_rank()
        self.sz = MPI.COMM_WORLD.Get_size()

    def test0(self):
        """
        Test 0-dimensional decomp
        """
        decomp = mumpy.CubeDecomp(self.sz, ())
        assert(not decomp.isValid())

    def test1(self):
        """
        Test 1-dimensional decomp
        """
        decomp = mumpy.CubeDecomp(self.sz, dims=(8,))
        if decomp.isValid():
            slab = decomp.getSlab(self.rk)
            for disp in (1,), (-1,):
                rk = decomp.getNeighborProc(self.rk, disp, periodic=[True])

    def test2(self):
        """
        Test 2-dimensional decomp
        """
        decomp = mumpy.CubeDecomp(self.sz, dims=(8, 16))
        if decomp.isValid():
            slab = decomp.getSlab(self.rk)
            for disp in (1, 0), (-1, 0), (1, -1):
                rk = decomp.getNeighborProc(self.rk, disp, periodic=[True, False])

    def test3(self):
        """
        Test 3-dimensional decomp
        """
        decomp = mumpy.CubeDecomp(self.sz, dims=(8, 16, 9))
        if decomp.isValid():
            slab = decomp.getSlab(self.rk)
            for disp in (0, 1, 0), (-1, 0, 0), (1, -1, 0), (1, -1, -1):
                rk = decomp.getNeighborProc(self.rk, disp, periodic=[True, False, True])

    def testInvalid(self):
        """
        Test invalid domain
        """
        decomp = mumpy.CubeDecomp(7, dims=(8, 16, 9))
        assert(not decomp.isValid())
       
if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCubeDecomp)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
    
