from itertools import permutations 
import numpy as np

class TSPGame():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    Use 1 for player1 and -1 for player2.
    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, args):
        self.num_node = args.num_node
        self.graph = np.random.rand(self.num_node, 2)

    def getStartState(self):
        """
        Returns:
            start_state: a representation of the graph
        """
        start_state = np.zeros([self.num_node, 2])
        start_state[0,0] = 1
        start_state[0,1] = 1
        return start_state

    def getActionSize(self):
        """
        Returns:
            self.num_node: number of all possible actions
        """
        return self.num_node

    def getNextState(self, state, action):
        """
        Input:
            state: current state
            action: action taken by current player
        Returns:
            next_state: graph after applying action
            reward: reward from action
        """
        next_state = state.copy()
        next_state[:, 1] = 0
        next_state[action, :] = 1
        prev_action = np.where(state[:, 1] == 1)[0][0]
        prev_node = self.graph[prev_action]
        cur_node = self.graph[action]
        reward = 1 - np.linalg.norm(cur_node - prev_node)
        if self.num_node == np.sum(next_state[:, 0]):
            reward += 1 - np.linalg.norm(cur_node - self.graph[0])
            
        return next_state, reward

    def getValidMoves(self, state):
        """
        Input:
            state: current state
        Returns:
            1 - state[:, 0]: a binary vector of length self.getActionSize(), 1 for
                            moves that are valid from the current board and player,
                            0 for invalid moves
        """
        return 1 - state[:, 0]

    def getGameEnded(self, state):
        """
        Input:
            state: current board state
        Returns:
            r: 0 if game has not ended. 1 if it has
               
        """
        r = 0
        if self.num_node == np.sum(state[:, 0]):
            r = 1
        return r

    def stringRepresentation(self, state):
        """
        Input:
            state: current state
        Returns:
            s: string representation of state
        """
        s = ''
        for i in range(self.num_node):
            s += str(int(state[i, 0]))
        return s
    
    def optimal_sol(self):
        """
        Input:
            
        Returns:
            optimal_val: optimal solution for TSP
            optimal_path: optimal path for TSP
        """
        cur_reward = 0
        optimal_val = float('inf')
        optimal_path = []
        graph = self.graph

        nodes = np.arange(self.num_node)[1:]
        perms = permutations(nodes)
        
        for perm in list(perms):
            cur_reward = 0
            
            cur_reward += np.linalg.norm(graph[0] - graph[perm[0]])
            for i in range(len(perm) - 1):
                j = perm[i]
                k = perm[i+1]
                cur_reward += np.linalg.norm(graph[k] - graph[j])
            cur_reward += np.linalg.norm(graph[perm[-1]] - graph[0])
            
            if optimal_val > cur_reward:
                optimal_val = cur_reward
                optimal_path = perm
        
        return optimal_val, optimal_path
    
    def create_sample(self):
        path = self.optimal_sol()[1]+tuple([0])
        current = self.getStartState()
        samples_v = []
        samples_pi = []
        tot_v = 0
        for i in path:
            pi = np.zeros(self.getActionSize())
            pi[i] = 1
            samples_pi.append((current, pi))
            next, v = self.getNextState(current, i)
            samples_v.append((current, v))
            current = next
        
        return samples_v, samples_pi
        
        