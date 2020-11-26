import time
import numpy as np
from scipy.linalg import block_diag
from consav.misc import elapsed, markov_rouwenhorst, choice
from consav import linear_interp

#Kode hentet ind
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
def asds(l,v_0):
    Na = 1000
    a = equilogspace(0,200,Na)
    new_v_0 = np.array_split(v_0,l)
    v_a = np.zeros(Na*l)
    for i in range(l):
        #new_v_0[0]
        # j = 1
        v_a[i*Na] = (new_v_0[i][1]-new_v_0[i][0])/(a[1]-a[0])

        # j inbetween
        for k in range((Na-1)):
            v_a[(1+k)+i*Na] = 0.5*((new_v_0[i][k+1]-new_v_0[i][k])/(a[k+1]-a[k]))+0.5*((new_v_0[i][k]-new_v_0[i][k-1])/(a[k]-a[k-1]))
            
        # j = last
        v_a[(Na-1)+i*Na]= (new_v_0[i][(Na-1)]-new_v_0[i][(Na-2)])/(a[(Na-1)]-a[(Na-2)])
    
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
    a_grid = equilogspace(0,200,Na)
    rho = 0.97 # AR(1) parameter
    sigma_e = 0.25 # std. of persistent shock
    sigma = 2
    beta = 0.96
    e_grid, e_trans, e_ergodic, e_trans_cumsum, e_ergodic_cumsum = markov_rouwenhorst(rho,sigma_e,Ne)

    #calculation
    #step 3
    A = np.kron(e_trans, np.identity(n))
    B = (beta*A)@v_a
    C = np.power(B,(-sigma))
    a_extend = np.tile(a_grid,11)
    m_endo = C + a_extend #samme opbygning som vektoren v_a

    #step 4
    values =[]
    m = (1+r)*a_grid[np.newaxis,:] + w*e_grid[:,np.newaxis]
    new_m_endo = np.array_split(m_endo,Ne)

    for k in range(Ne):
        linear_interp.interp_1d_vec(new_m_endo[k],a_grid,m[k],a[k])
        for i_a in range(Na):
            a[k,i_a] = np.fmax(a[k,i_a],0)
        values.append(m[k]-a[k])

    #print(f'new_m_endo[0]:{new_m_endo[0]}')
    #print(new_m_endo[0])
    c = np.concatenate(values)
    u = np.power(c,(1-sigma))/(1-sigma)
    u[u == -np.inf]=-1000000000000000000000000000
    print(f'c:{c}')
    print(f'u:{u}')

    #Create Q^
    lis = []
    Q = np.zeros(Ne*Na)
    #calculate
    for e in range(Ne):
        q = np.zeros((Na,Na)) #create each Q_i
        for k in range(Na):
            opt = a[e,k]
            if opt >= np.max(a_grid):
                q[k,Na-1] = 1 #If opt equals the end point

            elif opt <= np.min(a_grid):
                q[k,0] = 1 #If opt equals the start point

            else:   
                a_high = np.min(np.nonzero(opt<=a_grid)) #create points
                a_low = np.max(np.nonzero(opt>=a_grid))
                q[k,a_low] = (a_grid[a_low+1]-opt)/(a_grid[a_low+1]-a_grid[a_low])
                q[k,a_high] = (opt-a_grid[a_high-1])/(a_grid[a_high]-a_grid[a_high-1])
        lis.append(q)
        
    Q = block_diag(*lis) #make block matrix from Q's


    omega = Q@A
    
    new_v_a = np.linalg.inv(np.identity(11000)-beta*omega)@u
    return new_v_a



def loop(l, n, v_a):
    solve_tol = 0.01
    max_iter_solve = 1000
    it = 0
    while True:
        print(f'it:{it}')
        # i. save
        v_a_old = v_a.copy()
        #print(f'va_old:{v_a_old}')

        #step 2
        v_a_new =asds(l,v_a_old)
        #print(f'va_new:{v_a_new}')

        # step 7
        v_a = matrix(l, n, v_a_new)
        #print(f'v_a:{v_a}')
        # step 8
        print(f'max:{np.max(np.abs(v_a-v_a_old))}')
        if np.max(np.abs(v_a-v_a_old)) < solve_tol: break
        
        # iv. increment
        it += 1
        if it > max_iter_solve: raise Exception('too many iterations when solving for steady state')


