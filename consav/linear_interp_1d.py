import numpy as np
from numba import njit, boolean, int32, double, void
from .linear_interp import binary_search

@njit(double(double[:],double[:],double,int32),fastmath=True)
def _interp_1d(grid1,value,xi1,j1):
    """ 1d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        j1 (int): location in grid 

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right                    
        nom += nom_1*value[j1+k1]

    return nom/denom

@njit(double(double[:],double[:],double),fastmath=True)
def interp_1d(grid1,value,xi1):
    """ 1d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)

    return _interp_1d(grid1,value,xi1,j1)

@njit(void(double[:],double[:],double[:],double[:]),fastmath=True)
def interp_1d_vec(grid1,value,xi1,yi):
    """ 1d interpolation for vector of points
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (numpy.ndarray): input vector

    """

    for i in range(yi.size):
        yi[i] = interp_1d(grid1,value,xi1[i])

@njit(int32[:](int32),fastmath=True)
def interp_1d_prep(Nyi):
    """ preperation for 1d interpolation of only last dimension
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        Nyi (int): number of points to be evaluated

    Returns:

        prep (numpy.ndarray): information for remaining operations

    """

    # a. search in non-last dimensions
    
    # b. prep
    prep = np.zeros(Nyi,dtype=np.int32)

    return prep

@njit(void(int32[:],double[:],double[:],double[:],double[:],boolean,boolean),fastmath=True)
def _interp_1d_vec_mon(prep,grid1,value,xi1,yi,monotone,search):
    """ 1d interpolation for vector of points
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        yi (numpy.ndarray): output vector
        monotone (bool): indicator for whether xi1 is monotone
        search (bool): indicator for whether search is needed at all

    """

    # a. search in last dimension
    Nyi = yi.size
    if search:
        for i in range(Nyi):
            if monotone and i > 0:
                j1 = prep[i-1]
                while xi1[i] >= grid1[j1+1] and j1 < grid1.size-2:
                    j1 += 1
                prep[i] = j1
            else:
                prep[i] = binary_search(0,grid1.size,grid1,xi1[i])

    # b. interpolation 
    for i in range(yi.size):  
        yi[i] = 0.0 # initialize
        for k1 in range(2):
            j1 = prep[i]
            nom_1 = grid1[j1+1]-xi1[i] if k1 == 0 else xi1[i]-grid1[j1]            
            yi[i] += nom_1*value[j1+k1]

    for i in range(Nyi):
        j1 = prep[i]
        yi[i] /= (grid1[j1+1]-grid1[j1])

@njit(void(int32[:],double[:],double[:],double[:],double[:]),fastmath=True)
def interp_1d_vec_mon(prep,grid1,value,xi1,yi):
    """ 1d interpolation for vector of points where xi1 is monotone
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        yi (numpy.ndarray): output vector

    """

    _interp_1d_vec_mon(prep,grid1,value,xi1,yi,True,True)    

@njit(void(int32[:],double[:],double[:],double[:],double[:]),fastmath=True)
def interp_1d_vec_mon_rep(prep,grid1,value,xi1,yi):
    """ 1d interpolation for vector of points where xi1 is monotone and search is not needed
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        yi (numpy.ndarray): output vector

    """

    _interp_1d_vec_mon(prep,grid1,value,xi1,yi,True,False)       

@njit(void(double[:],double[:],double[:],double[:]),fastmath=True)
def interp_1d_vec_mon_noprep(grid1,value,xi1,yi):
    """ 1d interpolation for vector of points where xi1 is monotone
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        yi (numpy.ndarray): output vector

    """

    prep = interp_1d_prep(yi.size)
    _interp_1d_vec_mon(prep,grid1,value,xi1,yi,True,True)  
