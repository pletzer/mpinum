#/usr/bin/env python

"""
Distributed array class with ghosts

This code is provided with the hope that it will be useful. 
No guarantee is provided whatsoever. Use at your own risk.

See LICENSE.txt for conditions of use. 

Alex Pletzer (2012)
"""

import pnumpy
import numpy
from pnumpy import DistArray

__version__ = "$Id: pnGhostedDistArray.py 50 2016-05-27 11:12:29Z pletzer $"

class GhostedDistArray(pnumpy.DistArray):

    """
    Ghosted distributed array. Each process owns data and exposes the 
    halo region to other processes. These are accessed with tuples 
    such (1, 0) for north, (-1, 0) for south, etc.
    """

    def __init__(self, shape, dtype):
        """
        Constructor
        @param shape shape of the array
        @param dtype numpy data type
        @param numGhosts the width of the halo 
        """
        # call the parent Ctor
        DistArray.__init__(self, shape, dtype)

    def setNumberOfGhosts(self, numGhosts):
        """
	    Set the width of the ghost halo
	    @param numGhosts halo thickness
        """
        # expose each window to other PE domains
        ndim = len(self.shape)
        for dim in range(ndim):
            for drect in (-1, 1):
                # the window id uniquely specifies the
                # location of the window. we use 0's to indicate
                # a slab extending over the entire length for a
                # given direction, a 1 represents a layer of
                # thickness numGhosts on the high index side,
                # -1 on the low index side.
                winId = tuple( [0 for i in range(dim) ] \
                               + [drect] + \
                               [0 for i in range(dim+1, ndim) ] )
                
                slce = slice(0, numGhosts)
                if drect == 1:
                    slce = slice(self.shape[dim] - numGhosts, self.shape[dim])
        
                slab = self.getSlab(dim, slce)
                
                # expose MPI window
                self.expose(slab, winId)

    def getSlab(self, dim, slce):
        """
        Get slab. A slab is a multi-dimensional slice extending in
        all directions except along dim where slce applies
        @param dim dimension (0=first index, 1=2nd index...)
        @param slce python slice object along dimension dim
        @return slab
        """
        shape = self.shape
        ndim = len(shape)
        
        slab = [ slice(0, shape[i]) for i in range(dim) ] \
                    + [slce] + \
                  [ slice(0, shape[i]) for i in range(dim+1, ndim) ]
        return slab

    def getEllipsis(self, winID):
        """
        Get the ellipsis for a given halo side. 
        
        @param winID a tuple of zeros and one +1 or -1.  To access
               the "north" side for instance, set side=(1, 0),
               (-1, 0) to access the south side, (0, 1) the east
               side, etc. This does not involve any communication.

        @return None if halo was not exposed (bad winID)
        """
        if winID in self.windows:
            return self.windows[winID]['slice']
        else:
            return None

def ghArray(arry, dtype, numGhosts):
    """   
    ghosted distributed array constructor
    @param arry numpy-like array
    @param numGhosts the number of ghosts (>= 0)
    """ 
    a = numpy.array(arry, dtype)
    res = GhostedDistArray(a.shape, a.dtype)
    res.setNumberOfGhosts(numGhosts)
    res[:] = a # copy 
    return res

def ghZeros(shape, dtype, numGhosts):
    """
    ghosted distributed array zero constructor
    @param shape the shape of the array
    @param dtype the numpy data type 
    @param numGhosts the number of ghosts (>= 0)
    """
    res = GhostedDistArray(shape, dtype)
    res.setNumberOfGhosts(numGhosts)
    res[:] = 0
    return res

def ghOnes(shape, dtype, numGhosts):
    """
    ghosted distributed array one constructor
    @param shape the shape of the array
    @param dtype the numpy data type
    @param numGhosts the number of ghosts (>= 0)
    """
    res = GhostedDistArray(shape, dtype)
    res.setNumberOfGhosts(numGhosts)
    res[:] = 1
    return res

######################################################################
    
def test():

    import numpy

    # create local data container
    n, m = 2, 3

    # create dist array with 1 ghost
    
    da = ghZeros( (n, m), numpy.float64, 1 )
    rk = da.rk
    sz = da.sz

    # load the data on each PE
    data = numpy.reshape(numpy.array( [rk*100.0 + i for i in range(n*m) ] ),
                         (n,m))
    da[:] = data
    print(da)

    # this shows how one can access slabs 
    for pe in range(sz):
        winIndex = (-1, 0)
        print('[{0}] {1} slab belonging to {2} is: {3}\n'.format(rk, str(winIndex), pe), \
              da.get(pe, winID=winIndex))
        winIndex = (+1, 0)
        print('[{0}] {1} slab belonging to {2} is: {3}\n'.format(rk, str(winIndex), pe), \
              da.get(pe, winID=winIndex))
        winIndex = (0, -1)
        print('[{0}] {1} slab belonging to {2} is: {3}\n'.format(rk, str(winIndex), pe), \
              da.get(pe, winID=winIndex))
        winIndex = (0, +1)
        print('[{0}] {1} slab belonging to {2} is: {3}\n'.format(rk, str(winIndex), pe), \
              da.get(pe, winID=winIndex))

    # to keep mpi4py quite
    da.free()

if __name__ == '__main__': test()
