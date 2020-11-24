# -*- coding: utf-8 -*-
"""GEModelClass

Solves and simulates a buffer-stock consumption-saving problem for use in a general equilibrium model

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import njit, prange

# consav
from consav import ModelClass
from consav import linear_interp
from consav.misc import elapsed, equilogspace, markov_rouwenhorst, choice

############
# 2. model #
############

class GEModelClass(ModelClass):
    
    #########
    # setup #
    #########      

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. define list of non-float scalars
        self.not_float_list = ['Ne','Na','max_iter_solve','max_iter_simulate','simN']

        # b. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 0.96 # discount factor

        # c. prices
        par.r = 0.030 # interest rate
        par.w = 1.000 # wage rate

        # d. income parameters
        par.rho = 0.97 # AR(1) parameter
        par.sigma_e = 0.25 # std. of persistent shock
        par.Ne = 5 # number of states

        # f. grids         
        par.a_max = 200.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. misc.
        par.simN = 1_000_000 # number of households in MC simulation
        par.max_iter_solve = 5_000 # maximum number of iterations when solving
        par.max_iter_simulate = 5_000 # maximum number of iterations when simulating
        par.solve_tol = 1e-8 # tolerance when solving
        par.simulate_tol = 1e-4 # tolerance when simulating

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. grids
        self.create_grids()

        # a. solution
        sol_shape = (par.Ne,par.Na)
        sol.a = np.zeros(sol_shape)
        sol.m = np.zeros(sol_shape)
        sol.c = np.zeros(sol_shape)
        sol.V = np.zeros(sol_shape)
        sol.Va = np.zeros(sol_shape)

        # b. simulation
        sim_shape = (par.simN,)
        sim.i_e = np.zeros(sim_shape,dtype=np.int_)
        sim.a = np.zeros(sim_shape)
        sim.y = np.zeros(sim_shape)

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        par.a_grid = equilogspace(0,par.a_max,par.Na)
        par.e_grid, par.e_trans, par.e_ergodic, par.e_trans_cumsum, par.e_ergodic_cumsum = markov_rouwenhorst(par.rho,par.sigma_e,par.Ne)

    #########
    # solve #
    #########
    
    def solve(self,Va=None,do_print=False):
        """ solve the model """

        par = self.par
        sol = self.sol
        t0 = time.time()

        # a. create (or re-create) grids
        self.create_grids()

        # c. solve
        r = par.r
        w = par.w
        sol.m = (1+r)*par.a_grid[np.newaxis,:] + w*par.e_grid[:,np.newaxis]
        sol.a = 0.90*sol.m # pure guess
        sol.c = sol.m - sol.a 
        sol.Va = (1+r)*sol.c**(-par.sigma) if Va is None else Va
        sol.V = sol.c**(1-par.sigma)/(1-par.sigma)

        it = 0
        while True:

            # i. save
            a_old = sol.a.copy()
            V_old = sol.V.copy()

            # ii. solve backwards
            solve_backwards(par,r,w,sol.Va,sol.Va,sol.a,sol.c,sol.m,sol.V,sol.V)

            # iii. check
            if np.max(np.abs(sol.a-a_old)) < par.solve_tol: break
            
            # iv. increment
            it += 1
            if it > par.max_iter_solve: raise Exception('too many iterations when solving for steady state')

        if do_print:
            print(f'household problem solved in {elapsed(t0)} [{it} iterations]')
            print(f'[value function abs. max diff. is {np.max(np.abs(sol.V-V_old)):.8f}]')

    def simulate(self,seed=1917,do_print=False):
        """ simulate the model """
        
        par = self.par
        sol = self.sol
        sim = self.sim        
        t0 = time.time()

        # a. initial guess
        r = par.r
        w = par.w

        if not seed is None: np.random.seed(seed)
        sim.i_e = np.random.choice(par.Ne,p=par.e_ergodic,size=(par.simN,))
        sim.a = np.zeros(par.simN)

        # b. simulate
        it = 0
        while True:

            # i. save
            a_old = sim.a.copy()

            # ii. simulate forward 
            simulate_forwards(par,sol,r,w,sim.i_e,sim.y,sim.a)

            # iii. check
            diff_mean = np.abs(np.mean(sim.a)-np.mean(a_old))
            diff_std = np.abs(np.std(sim.a)-np.std(a_old))
            if np.fmax(diff_mean,diff_std) < par.simulate_tol: break
            
            # iv. increment
            it += 1
            if it > par.max_iter_simulate: raise Exception('too many iterations when simulating towards steady state')

        if do_print:
            print(f'household problem simulated in {elapsed(t0)} [{it} iterations]')

######################
# fast jit functions #
######################

@njit(parallel=True)        
def solve_backwards(par,r,w,Va_p,Va,a,c,m,V_p,V):
    """ perform time iteration step with Va_p from previous iteration """

    # a. post-decision
    marg_u_plus = (par.beta*par.e_trans)@Va_p
    Vbar = (par.beta*par.e_trans)@V_p
        
    # convert from domain ]-\infty:0[ to ]0:infty[
    # -> extrapolation does not break domain does

    inv_Vbar = -1.0/Vbar 

    # b. egm loop
    for i_e in prange(par.Ne):
        
        # i. egm
        c_endo = marg_u_plus[i_e]**(-1/par.sigma)
        m_endo = c_endo + par.a_grid

        # ii. interpolation
        linear_interp.interp_1d_vec(m_endo,par.a_grid,m[i_e],a[i_e])
        a[i_e,0] = np.fmax(a[i_e,0],0)
        c[i_e] = m[i_e]-a[i_e]

        # iii. envelope condition
        Va[i_e] = (1+r)*c[i_e]**(-par.sigma)

        # iv. value function
        for i_a in range(par.Na):
            inv_Vbar_now = linear_interp.interp_1d(par.a_grid,inv_Vbar[i_e],a[i_e,i_a])
            Vbar_now = -inv_Vbar_now # convert back
            V[i_e,i_a] = c[i_e,i_a]**(1-par.sigma)/(1-par.sigma) + Vbar_now

@njit(parallel=True)        
def simulate_forwards(par,sol,r,w,i_e,y,a):
    """ simulate forward """

    for i in prange(par.simN):

        # i. update income
        i_e_lag = i_e[i]
        pi_e_val = np.random.uniform(0,1)
        i_e[i] = choice(pi_e_val,par.e_trans_cumsum[i_e_lag,:])
        y[i] = w*par.e_grid[i_e[i]]

        # ii. update assets
        a[i] = linear_interp.interp_1d(par.a_grid,sol.a[i_e[i]],a[i])        