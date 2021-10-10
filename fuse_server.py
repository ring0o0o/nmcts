# coding:utf-8

from __future__ import division

import numpy as np
import time
import math
import random
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from copy import deepcopy 

def pearson_coef_filter(a,b):
    Cp = 0.9
    x = []
    y = []
    for i in range(a.shape[1]):
        x.append(np.corrcoef(a.iloc[:,i],b,rowvar = False)[0,1])
    x = np.asarray(x)
    y = np.where((x>Cp)|(x<-Cp))
    if len(y[0]):
        return False
    else:
        return True

class treeNode():
    def __init__(self, state, parent):
        self.children = {}
        self.state = state
        self.isTerminal = state.isTerminal()
        self.parent = parent
        self.numVisits = 0
        self.rolloutReward = 0
        self.mReward = 0
        self.expand_width = 10

    def isFullyExpanded(self):
        if len(self.children) == self.expand_width:
            return True
        else:
            return False

class ReliefF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, k=10, sigma=30):
        """Sets up ReliefF to perform feature selection.
        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.
        Returns
        -------
        None
        """

        self.k = k
        self.sigma = sigma

    # This function finds the k nearest neighbours
    def knnsearchR(self,A, b, n):
        difference = (A - b)**2
        sumDifference = np.sum(difference, axis = 1)
        neighbourIndex = np.argsort(sumDifference)
        neighbours = A[neighbourIndex][1:]
        knn = neighbours[:n]
        return knn, neighbourIndex[1:] #Don't want to count the original point

    # This follows the Eqn 8
    def distance(self):
        d1 = [np.exp(-((n + 1) / self.sigma) ** 2) for n in range(self.k)]
        d = d1 / np.sum(d1)
        return d

    # This follows Eqn 2
    def diffNumeric(self,A, XRandomInstance, XKNNj, X):
        denominator = np.max(X[:, A]) - np.min(X[:, A])
        return np.abs(XRandomInstance[A] - XKNNj[A]) / denominator

    def getWeight(self, X, y, updates='all'):
        # Check if user wants all values to be considered
        if updates == 'all':
            m = X.shape[0]
        else:
            m = updates

        # The constants need for RReliefF
        N_dC = 0
        N_dA = np.zeros([X.shape[1],1])
        N_dCanddA= np.zeros([X.shape[1],1])
        W_A = np.zeros([X.shape[1],1])
        Wtrack = np.zeros([m, X.shape[1]])
        yRange = np.max(y) - np.min(y)
        iTrack = np.zeros([m,1])

        # Repeat based on the total number of inputs or based on a user specified value
        for i in range(m):

            # Randomly access an instance
            if updates == 'all':
                random_instance = i
            else:
                random_instance = np.random.randint(low=0, high=X.shape[0])

            # Select a 'k' number in instances near the chosen random instance
            XKNN, neighbourIndex = self.knnsearchR(X, X[random_instance,:],self.k)
            yKNN = y[neighbourIndex]
            XRandomInstance = X[random_instance, :]
            yRandomInstance = y[random_instance]

            # Loop through all selected random instances
            for j in range(self.k):

                # Weight for different predictions
                N_dC += (np.abs(yRandomInstance-yKNN[j])/yRange) * self.distance()[j]

                # Loop through all attributes
                for A in range(X.shape[1]):

                    # Weight to account for different attributes
                    N_dA[A] = N_dA[A] +  self.diffNumeric(A, XRandomInstance, XKNN[j], X) * self.distance()[j]

                    # Concurrent examination of attributes and output
                    N_dCanddA[A] = N_dCanddA[A] + (np.abs(yRandomInstance-yKNN[j])/yRange) * self.distance()[j] *\
                                   self.diffNumeric(A, XRandomInstance, XKNN[j], X)

            # This is another variable we use to keep track of all weights - this can be used to see how RReliefF works
            for A in range(X.shape[1]):
                Wtrack[i, A] = N_dCanddA[A] / N_dC - ((N_dA[A] - N_dCanddA[A]) / (m - N_dC))

            # The index corresponding to the weight
            iTrack[i] = random_instance

        # Calculating the weights for all features
        for A in range(X.shape[1]):
            W_A[A] = N_dCanddA[A]/N_dC - ((N_dA[A]-N_dCanddA[A])/(m-N_dC))

        return abs(W_A.ravel())

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, Cp = 0.9, explorationConstant=0.1,
                 X=None,y=None):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.Cp = Cp
        self.explorationConstant = explorationConstant
        self.X = X
        self.y = y
        self.level = {}

        

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
            self.bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.bestChild)

    def executeRound(self):
        node = self.treePolicy(self.root)
        m = node.state.feature_subset        
        if len(m) not in self.level.keys():
            self.level[len(m)] = [node]
        else: self.level[len(m)].append(node)

        rolloutReward = self.defaultPolicy(node.state)

        mReward = rolloutReward[:len(m)]
        self.backpropogate(node, mReward)

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        if explorationValue:
            for child in node.children.values():
                nodeValue = child.mReward + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
                #nodeValue  = TSC(node,nodeValue)
                if nodeValue > bestValue:
                    bestValue = nodeValue
                    bestNodes = [child]
                elif nodeValue == bestValue:
                    bestNodes.append(child)
            return random.choice(bestNodes)
        else:
            for k,v in self.level.items():
                for i in v:
                    nodeValue = i.mReward
                    if nodeValue > bestValue:
                        bestValue = nodeValue
                        bestNodes = [i]
                    elif nodeValue == bestValue:
                        bestNodes.append(i)                    
            return np.random.choice(bestNodes)

            # for child in node.children.values():
            #     if child is None:
            #         print('node has no children')
            #     nodeValue = child.mReward
            #     if nodeValue > bestValue:
            #         bestValue = nodeValue
            #         bestNodes = [child]
            #     elif nodeValue == bestValue:
            #         bestNodes.append(child)
            # return random.choice(bestNodes)
        #return bestNodes

    def defaultPolicy(self,state):
        while len(state.feature_subset) < state.search_depth and len(state.feature_subset) < len(state.Feature_COLUMNS):
            try:
                action = random.choice(state.getPossibleActions(state.feature_subset))
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.takeAction(action)
        return state.getReward(self.X[state.feature_subset],self.y)

    def treePolicy(self, node):
        while not node.isTerminal:
            if not node.isFullyExpanded():
                return self.expand(node)
            else:
                node = self.getBestChild(node, self.explorationConstant)
                print('bestChild is',node.state.feature_subset)
        return node

    def expand(self, node):
        filtered_actions = []
        exclude_list = []
        if node.parent in self.root.children:
            exclude_list = list(self.root.children.keys())
            exclude_list.append(node.children.keys())
        else:
            exclude_list=list(node.children.keys())
        actions = node.state.getPossibleActions(exclude_list)
        if node is not self.root:
            for action in actions:
                filtered_actions.append(pearson_coef_filter(self.X.loc[:,node.state.feature_subset],self.X.loc[:,action] ))
            actions = np.asarray(actions)
            action = np.random.choice(actions[filtered_actions])
        else: action  = np.random.choice(actions[:node.expand_width])
        newNode = treeNode(node.state.takeAction(action), node)
        node.children[action] = newNode
        newNode.mReward = 0.0
        newNode.numVisits = 0
        return newNode

    def backpropogate(self, node, reward):
        t = 0
        #print('feature_subset',node.state.feature_subset)
        while node is not self.root:
            t += 1
            node.numVisits += 1
            if node.parent is self.root:
                node.parent.numVisits += 1
            node.mReward += (reward[-t] - node.mReward ) / node.numVisits
            node = node.parent

    def getAction(self, bestChild):
        return bestChild.state.feature_subset


