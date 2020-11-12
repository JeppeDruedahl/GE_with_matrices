import time
import numpy as np
from numba import njit, prange
from consav.misc import elapsed, markov_rouwenhorst, choice
from consav import linear_interp


def equilogspace(x_min,x_max,n):
    """ like np.linspace. but (close to)  equidistant in logs

    Args:

        x_min (double): maximum value
        x_max (double): minimum value
        n (int): number of points
    
    Returns:

        y (list): grid with unequal spacing

    """

    pivot = np.abs(x_min) + 0.25
    y = np.geomspace(x_min + pivot, x_max + pivot, n) - pivot
    y[0] = x_min  # make sure *exactly* equal to x_min
    return y


#######
####### Christoffers kode
#######

### Make v_a ####
def asds(l):
    a = equilogspace(0,200,1000)
    v_0 = np.arange(l*1000)
    new_v_0 = np.array_split(v_0,l)
    v_a = np.zeros(1000*l)
    for i in range(l):
        #new_v_0[1]
        # j = 1
        v_a[i*1000] = (new_v_0[i][1]-new_v_0[i][0])/(a[1]-a[0])

        # j inbetween
        for k in range(999):
            v_a[k+i*1000] = 0.5*((new_v_0[i][k+1]-new_v_0[i][k])/(a[k+1]-a[k]))+0.5*((new_v_0[i][k]-new_v_0[i][k-1])/(a[k]-a[k-1]))

        # j = last
        v_a[999+i*1000]= (new_v_0[i][999]-new_v_0[i][998])/(a[999]-a[998])
    
    return v_a
    

### Calculation of v_a^{n+1}
def matrix(l,n,v_a):

    #parameter
    Na = 1000
    Ne = 11 # number of states
    sol_shape = (Ne,Na)
    a = np.zeros(sol_shape)
    r = 0.03
    w = 1.000 
    a_grid = equilogspace(0,200,1000)
    rho = 0.97 # AR(1) parameter
    sigma_e = 0.25 # std. of persistent shock
    sigma = 2
    beta = 0.96
    e_grid, e_trans, e_ergodic, e_trans_cumsum, e_ergodic_cumsum = markov_rouwenhorst(rho,sigma_e,Ne)

    #calculation
    #step 3
    A = np.kron(e_trans, np.identity(n))
    C = (beta*A)@v_a
    B = np.power(C,(-sigma))
    a_extend = np.repeat(a_grid,11)
    m_endo = B + a_extend #samme opbygning som vektoren v_a
    print(A,C[900]) 
    print(B[900],m_endo)

    #step 4
    values =[]
    m = (1+r)*a_grid[np.newaxis,:] + w*e_grid[:,np.newaxis]
    new_m_endo = np.array_split(m_endo,Ne)

    for k in range(Ne):
        linear_interp.interp_1d_vec(new_m_endo[k],a_grid,m[k],a[k])
        a[k,0] = np.fmax(a[k,0],0)
        values.append(a[k]-m[k])

    print(new_m_endo[0])
    c = np.concatenate(values)
    u = np.power(c,(1-sigma))/(1-sigma)

    #Create Q
    #Start med e=0
    q = np.zeros((Na,Na))
    for k in range(Na):
        opt = a[0,k]
        #print(opt)
        a3 = a_grid[opt<=a_grid].argmax()
        a4 = a_grid[opt>=a_grid].argmin()
        q[k,a3] = (opt-a_grid[a3-1])/(a_grid[a3]-a_grid[a3-1])
        q[k,a4] = (opt-a_grid[a4])/(a_grid[a4+1]-a_grid[a4])
        #print(k) 
    
    #print(q[100])


