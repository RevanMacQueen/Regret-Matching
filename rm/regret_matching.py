import numpy as np
from rm.utils import expected_utility
from itertools import product
import scipy.optimize as op
import copy


class RegretMatching():
    def __init__(self, i, num_actions, payoffs, blue_print=None):
        self.cumulative_regrets = np.zeros(num_actions) 
        self.i = i
        
        if blue_print is None:
            self.strategy = np.ones(num_actions)/num_actions 
            self.blue_print = None
        else:
            self.strategy = blue_print
            self.blue_print = blue_print

        self.payoffs = payoffs # payoff matrix for this player

        self.history = None # history of stategy profiles. Used to compute regret
        self.num_actions = num_actions
        self.t = 0
        self.cumulative_strategy = np.zeros(num_actions) # average strategy


    def get_action(self):
        a = np.random.choice(list(range(self.num_actions)), p=self.strategy)
        s_i = np.zeros(self.num_actions)
        s_i[a] = 1
        return s_i

    def get_average_strategy(self):
        return  self.cumulative_strategy /self.t  

    def update(self, s):
        '''
        Updates at each iteration. 
            First updates history
            Then computes regret given deviation type
            Then defines new strategy

        parameters:
            s : (np.array) strategy profile at last iteration

        see: http://anytime.cs.umass.edu/aimath06/proceedings/P47.pdf
        '''

        self.cumulative_strategy += self.strategy
        self.t += 1
        
        for a in range(self.num_actions):   #just handle external regret for now
            s_i = np.zeros(self.num_actions)
            s_i[a] = 1.0
            s_t_prime = copy.deepcopy(s)
            s_t_prime[self.i] = s_i
            self.cumulative_regrets[a] +=  self.expected_utility(s_t_prime) - self.expected_utility(s) # this uses the observed regret

        y = self.link(self.cumulative_regrets) 
    
        if np.sum(y) != 0:
            M = np.zeros((self.num_actions, self.num_actions))
            for a in range(self.num_actions):
                s_i = np.zeros(self.num_actions)
                s_i[a] = 1.0
                phi = np.repeat(np.expand_dims(s_i, axis=0), [self.num_actions], axis=0)
                M += phi*y[a]
            M = M / np.sum(y)
            self.strategy  = op.fixed_point(lambda x: x.dot(M), x0=self.strategy)
            assert np.all(np.isclose(self.strategy, y/np.sum(y)))

        else:
            if self.blue_print is None:
                self.strategy = np.ones(self.num_actions)/self.num_actions 
            else:
                self.strategy = self.blue_print

    def link(self, regrets):
        return np.maximum(regrets, np.zeros_like(regrets))

    def get_regret(self, s):
        regret = np.zeros(self.num_actions)

        for a in range(self.num_actions):   #just handle external regret for now
            s_i = np.zeros(self.num_actions)
            s_i[a] = 1.0
            s_t_prime = copy.deepcopy(s)
            s_t_prime[self.i] = s_i
            regret[a] =  self.expected_utility(s_t_prime) - self.expected_utility(s) # this uses the observed regret
        return regret

    def compute_external_regret(self, s_i):
        '''
        computes external regret, i.e. what if s_i was played in all previous games
        '''
        T = self.history.shape[0] # current time
        regret  = 0
        for t in range(T): # all previous timesteps
            s_t = self.history[t] # action profile that actually happened
            s_t_prime = self.history[t].copy()
            s_t_prime[self.i] = s_i
            regret +=  self.expected_utility(s_t_prime) - self.expected_utility(s_t)
        
        return regret/T # returns average regret


    def expected_utility(self, s):
        '''
        Gets the expected utilitiy for self under strategy profile s
        
        Parameters
            s : (list) list where S[i] gives the probability of player i playing each action

        Returns
            eu : (np.array) expected utility for each player  

        '''
        return expected_utility(s, self.payoffs)



class InternalRegretMatching():
    def __init__(self, i, num_actions, payoffs,  blue_print=None):
        self.i = i
        self.deviation_set = self.get_deviation_set(num_actions)
       
        if blue_print is None:
            self.strategy = np.ones(num_actions)/num_actions 
            self.blue_print = None
            self.cumulative_regrets = np.zeros(len(self.deviation_set))

        else:
            self.strategy = blue_print
            self.blue_print = blue_print

            self.cumulative_regrets = np.ones(len(self.deviation_set))


        self.avg_strat = np.zeros(num_actions) # average strategy
        self.payoffs = payoffs # payoff matrix for this player
        self.num_actions = num_actions
        self.t = 0

        
    def get_deviation_set(self, num_actions):
        '''
        Returns a list of internal deviations in matrix form

        parameters:
            num_actions : (int) the number of actions for this player
        returns :
            deviation_set : (list) list of np.arrays representing interal action transformations  
        '''

        actions = list(range(num_actions))
        mappings = product(actions, actions) # all the different mappings from A_i -> A_i
        deviation_set = [np.identity(num_actions)] # include identity mapping
        for phi in mappings:
            a = phi[0] # the orginal action
            phi_a = phi[1] # the transformed action under phi
            if a != phi_a: # pass if the deviation maps the action to itself
                phi_mat = np.identity(num_actions)
                phi_mat[a, a] = 0
                phi_mat[a, phi_a] = 1
                deviation_set.append(phi_mat)

        assert len(deviation_set) == num_actions**2 - num_actions + 1
        return deviation_set


    def get_action(self):
        a = np.random.choice(list(range(self.num_actions)), p=self.strategy)
        s_i = np.zeros(self.num_actions)
        s_i[a] = 1
        return s_i


    def get_average_strategy(self):
        return  self.avg_strat/self.t


    def update(self, s):
        '''
        Updates at each iteration. 
            First updates history
            Then computes regret given deviation type
            Then defines new strategy

        parameters:
            s : (np.array) strategy profile at last iteration

        see: http://anytime.cs.umass.edu/aimath06/proceedings/P47.pdf
        '''

        self.avg_strat += self.strategy
        self.t += 1
        
        for phi_idx, phi in enumerate(self.deviation_set):
            phi_a = np.matmul(s[self.i], phi)
            s_t_prime = copy.deepcopy(s)
            s_t_prime[self.i] = phi_a            
            self.cumulative_regrets[phi_idx] +=  self.expected_utility(s_t_prime) - self.expected_utility(s) 

        y = self.link(self.cumulative_regrets) 
    
        if np.sum(y) != 0:
            M = np.zeros((self.num_actions, self.num_actions))
            for phi_idx, phi in enumerate(self.deviation_set):
                M += phi*y[phi_idx]

            M = M / np.sum(y)
           
            evals, evecs = np.linalg.eig(M.T)
            evec1 = evecs[:,np.isclose(evals, 1)]

            #Since np.isclose will return an array, we've indexed with an array
            #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
            evec1 = evec1[:,0]

            stationary = evec1 / evec1.sum()

            #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
            self.strategy = stationary.real

            if not np.all(self.strategy>=0):
                self.strategy[self.strategy<0] = 0
        else:
            if self.blue_print is None:
                self.strategy = np.ones(self.num_actions)/self.num_actions 
            else:
                self.strategy = self.blue_print


    def link(self, regrets):
        return np.maximum(regrets, np.zeros_like(regrets))


    def expected_utility(self, s):
        '''
        Gets the expected utilitiy for self under strategy profile s
        
        Parameters
            s : (list) list where S[i] gives the probability of player i playing each action

        Returns
            eu : (np.array) expected utility for each player  

        '''
        return expected_utility(s, self.payoffs)