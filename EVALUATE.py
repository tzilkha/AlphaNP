import numpy as np
import pandas as pd
from MCTS import MCTS
import matplotlib.pyplot as plt
import copy

# Given a dictionary of MCTSs and a Game, returns a dictionary of each models best results over number of simulations
def evaluation_run(game, MCTSs):
    results = {}
    optimal_val, optimal_path = game.optimal_sol()
    for model in MCTSs.keys():
        state = [game.getStartState()]
        mcts_reward = 0
        mcts_actions = []
        
        while not game.getGameEnded(state[-1]):
            action = np.argmax(MCTSs[model].getActionProb(state))
            next_state, reward = game.getNextState(state[-1], action)
            state+=[next_state]
            mcts_actions += [action]
            mcts_reward += reward
        results[model] = copy.deepcopy((MCTSs[model].val, MCTSs[model].num_sim))
    return results, optimal_val   

# Given MCTS results and optimal solution returns a pandas series of the commulative number of games where a solution
# within a threshold of the optimal was found per number of simulations
def add_entry(model_results, optimal, args):
    count_v = np.zeros(args['numMCTSSims']+1)
    for i,j in enumerate(model_results[1]): count_v[j] = model_results[0][i]
    return ((pd.Series(count_v).replace(to_replace=0, method='ffill')/optimal)<1.1)*1

# Given games and neural nets, runs models on games and computes the number of games which a solution was found
# within a threshold of the optimal
#
# games - a list of games to test on the models
# nets - a dictionaryu of the neural nets to be tested, None for no net
# args - argument dictionary
def create_comparison(games, nets, args):
    totals = {nn:pd.Series(np.zeros(args['numMCTSSims']+1)) for nn in nets.keys()}
    for i,game in enumerate(games):
        print(i, end='...')
        mcts_dic = {nn:MCTS(game, nets[nn], args) for nn in nets.keys()}
        results, optimal = evaluation_run(game, mcts_dic)
        for model in totals.keys():
            totals[model] += (add_entry(results[model], optimal, args))
    for model in totals.keys():
        totals[model]/=len(games)
    print()
    return totals

# Prints results from create_comparison
def plot_comparison(res):
    for model in res.keys():
        plt.plot(list(range(len(res[model]))), res[model])
    plt.xlabel("Number of Simulations")
    plt.ylabel("Percentage of Games within Threshold")
    plt.legend(res.keys())
    plt.show()