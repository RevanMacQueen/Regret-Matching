'''Utility functions relating to equilibria'''

import numpy as np

from copy import deepcopy
from scipy.stats.contingency import margins
from rm.regret_matching import RegretMatching


def compute_best_response(s, player_i, iterations=10000):
    '''
    Computes best response of a player i to a fixed strategy profile s using regret matching.  

    parameters:
        s : (list) strategy 
        i : (agents.RegretMatching) the player for which to compute a best response

    returns:
       (np.array) best reponse for player i 
    '''

    # set up regret matching player with the same attributes as i
    i = player_i.i 
    num_actions = player_i.num_actions
    payoffs = player_i.payoffs
    best_responder = RegretMatching(i, num_actions, payoffs)

    # traing using regret matching
    for itr in range(iterations):
        a = best_responder.get_action()
        s[i] = a
        best_responder.update(s)

    return  best_responder.get_average_strategy()


def is_epsilon_nash(s, players, epsilon, iterations=1000):
    '''
    Check to see if a given strategy profile is an epsilon nash 
    
    uses algorithm from: https://poker.cs.ualberta.ca/publications/AAMAS10.pdf

    parameters:
        s : (list) strategy 
        players : (list) list of regret matching players
        epsilon : (float) tolerance for epsilon-nash

    return:
        (bool) : whether this is an epsilon nash equilibrium
    '''
    num_players = len(players)

    # compute expected utility under s and s_br for each player
    eu_s = np.zeros(num_players)
    eu_s_br = np.zeros(num_players)

    for i in range(num_players):
        eu_s[i] = players[i].expected_utility(s)
        # s_br_i is a strategy profile where only i is best responding
        s_br_i = deepcopy(s) 
        s_br_i[i] = compute_best_response(deepcopy(s), players[i], iterations=iterations)
        eu_s_br[i] = players[i].expected_utility(s_br_i)

    avg_diff =  np.mean(eu_s_br-eu_s)
    if  avg_diff<= epsilon:
        return True, avg_diff
    else:
        return False, avg_diff


def get_marginal_strategy(d, player):
    '''
    Given a joint distribution over action profiles d, returns player's marginal distribution over
    action.

    parameters:
        d : (np.array) array with dimension |A| where A is the set of action profiles
        player : (int) integer index of the player
    returns:
        marginal_s : (np.array) array with |A_i| entries; the marginal probability of each action for i
    '''
    marginal_d = margins(d)
    marginal_s = marginal_d[player].flatten()
    return marginal_s


def is_epsilon_coarse_correlated(d, payoffs, num_actions, epsilon):
    '''
    Checks if d, a probability distribution over action profiles, is an epsilon coarse correlated eq

    parameters:
        d : (np.array) the probability dist.; array with dimension |A| where A is the set of action profiles, 
        payoffs : (np.array); array with dimension |A|+1 where A is the set of action profiles
        num_actions : (list) list where num_actions[i] gives the number of actions for player i
        epsilon : (float) error term
    returns:
        is_coarse_correlated : (bool) whether d is an epsilon-CCE
        max_deviation_payoff : (np.array) array of size (num_players), with the maxmimum benefit for deviating
    '''
    num_players = len(num_actions)
    max_deviation_payoff = np.ones(num_players)*-np.inf # the maximum payoff for deviation for each player
    is_coarse_correlated = True

    for i in range(num_players):
        d_no_i = np.sum(d, axis=i) #joint distribution over A_{-i}, by marginalizing out i
        eu_d = np.dot(payoffs[i].flatten(), d.flatten()) # expected utility for i under d

        for a in range(num_actions[i]):
            # s_i = np.zeros(num_actions[i])
            # s_i[a] = 1
            actions = [list(range(num_actions[j])) for j in range(num_players)]
            actions[i] = [a]            

            idx = np.ix_(*actions)
            payoffs_given_a = payoffs[i] 
            payoffs_given_a = np.squeeze(payoffs_given_a[idx]) # payoffs for i, given they play a and other players play according to d
            
            eu_deviate_a = np.sum(np.multiply(d_no_i, payoffs_given_a))
            deviation_payoff = eu_deviate_a - eu_d 
            
            if deviation_payoff >= max_deviation_payoff[i]:
                max_deviation_payoff[i] = deviation_payoff

            if deviation_payoff >= epsilon: # i
                is_coarse_correlated = False
                

    return is_coarse_correlated, max_deviation_payoff