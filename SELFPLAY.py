from collections import deque
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
import copy
import pandas as pd

from TSP import TSPGame
from MCTS import MCTS

class Coach():
    
    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False    # can be overriden in loadTrainExamples()
        self.game = None
        self.mcts = None
        self.gen = 1

    def executeEpisode(self):
        self.game = TSPGame(self.args)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        
        trainExamples = []
        board = [self.game.getStartState()]
        episodeStep = 0

        while True:
            episodeStep += 1

            pi = self.mcts.getActionProb(board)

            action = np.random.choice(len(pi), p=pi)
            next_board, reward = self.game.getNextState(board[-1], action)
            
            trainExamples.append([copy.deepcopy(board), self.game.graph, pi, reward])
            
            board.append(next_board)
            
            r = self.game.getGameEnded(board[-1])
            
            if r!=0:
                return [tuple(x) for x in trainExamples]
            
    def learn(self):
        trainexamples = []

        for eps in range(self.args.numEps):
            trainexamples += self.executeEpisode()
    
        shuffle(trainexamples)
            
        return trainexamples

# Old Arena style self-play
#
#     def next_gen(self):
#         games = [TSPGame(self.args) for i in range(self.args['numAssess'])]
#         self.nnet.save_model('Generation_'+str(self.gen))
        
#         print("Assessing old...")
        
#         results = {"old":[], "new":[]}
#         for g in games:
#             results['old'] += [self.play_game(self.nnet, g)]
         
#         examples = self.learn()
        
#         print("Dataset Size", len(examples))
        
#         self.nnet.train(examples)
        
#         print("Assessing new...")
        
#         for g in games:
#             results['new'] += [self.play_game(self.nnet, g)]  
            
#         print("\n################################### - RESULTS - ###################################")
#         res = ["Old" if results['new'][i]<results['old'][i] else("New" if results['new'][i]>results['old'][i] else "Tie")  \
#                for i in range(len(results['new']))]
#         print(pd.Series(res).value_counts())

#         counts = {'New':0, 'Old':0, 'Tie':0}
#         for key in counts.keys():
#             counts[key] = res.count(key)
        
#         if (counts['Old']==0) or (counts['New']/counts['Old']>self.args['winThresh']):
#             print("NEW GENERATION SUPERIOR")
#             print("Updating model.")
#             self.gen += 1
#         else:
#             print("New Generation INFERIOR")
#             print('Rejecting model.')
#             self.nnet.load_model('Generation_'+str(self.gen))
            
#         print("###################################################################################\n")
        
#         return counts
        
    
    def next_gen(self):
#         games = [TSPGame(self.args) for i in range(self.args['numAssess'])]
        trainex = []
        naive_res = []
        mcts_res = []
        totalimprov = 0
        print("Creating samples...")
        for i in range(self.args['numAssess']):
            g = TSPGame(self.args)
            naive_res += [self.naive_play(self.nnet, g, self.args['history'])[1]]
            playgame = self.play_game(self.nnet, g)
            mcts_res += [playgame[1]]
            trainex += [playgame[0]]
        for i in range(len(naive_res)):
            if naive_res[i] < mcts_res[i]:
                totalimprov += 1
                self.trainExamplesHistory += trainex[i]
        print(totalimprov, "New Games Generated.")
        if len(self.trainExamplesHistory) > self.args['maxTrain']:
            self.trainExamplesHistory = self.trainExamplesHistory[-self.args['maxTrain']:]
            print("Samples deleted.")       
                                                                  
        print("Training on", len(self.trainExamplesHistory), "samples...")
        res = self.nnet.train(self.trainExamplesHistory)
        self.nnet.save_model("BestGen")
        return res                                                          
        
        
        
        
        
        
        
    def naive_play(self, nn, game, with_hist = False):
        state = [game.getStartState()]
        reward = 0
        history = []
        while not game.getGameEnded(state[-1]):
            if with_hist: pi = nn.predict(state, game.graph)[0]
            else: pi = nn.predict(state[-1], game.graph)[0]
                
            valid = game.getValidMoves(state[-1])
            action = np.argmax(pi)
            while valid[action]==0:
                pi[action]=0
                if np.sum(pi)==0:
                    print("Bad pi - random policy.")
                    pi = np.random.rand(game.num_node)
                action = np.argmax(pi)
            next_state, r = game.getNextState(state[-1], action)
            history += [[copy.deepcopy(state), game.graph, pi/np.sum(pi), r]]    
            
            state += [next_state]
            reward += r
        return history, reward
            
    
    def play_game(self, nn, game):
        mcts = MCTS(game, nn, self.args)
        
        state = [game.getStartState()]
        mcts_reward = 0
        history = []
        while not game.getGameEnded(state[-1]):
            action = np.argmax(mcts.getActionProb(state))
            next_state, reward = game.getNextState(state[-1], action)
            state.append(next_state)
            ac = np.zeros(game.getActionSize())
            ac[action] = 1
            history += [[copy.deepcopy(state), game.graph, ac, reward]]
            mcts_reward += reward
        return history, mcts_reward

    def arena(self):
        counts = []
        
        for i in range(self.args['numGens']):
            print("Generation", i)
            counts += [self.next_gen()]
            print()
        
        return counts