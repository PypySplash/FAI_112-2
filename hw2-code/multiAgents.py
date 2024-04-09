# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()  # Returns a Grid of boolean food indicator variables.
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(newGhostStates)
        # 初始化評分
        score = 0
        
        # 考慮食物的距離
        foodList = newFood.asList()
        if foodList:  # 檢查 foodList 是否非空
            # 考慮最近食物
            # closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            # # score -= closestFoodDist  # 透過用減的，可以達到: 距離越近，分數越高
            # score += 1.0 / closestFoodDist  # 使用倒數，距離越近分數增幅越大
            
            # 考慮所有食物的位置
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDist = min(foodDistances)
            score += 2.0 / closestFoodDist  # 加大最近食物的權重
            score += sum(1.0 / (dist + 1e-5) for dist in foodDistances) / len(foodDistances)  # 加入平均倒數距離作為一個因素  # 避免除以零
            # score += sum([1.0 / dist for dist in foodDistances])  # 考慮所有食物的倒數距離
        
        # 考慮鬼魂的位置和驚嚇狀態
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            # 動態計算與鬼魂的安全距離
            safeDistance = 2 + (scaredTime > 0) * 2  # 每個鬼魂的 scaredTime 可能不同，需在循環內計算
            
            if ghostDistance < safeDistance:
                if scaredTime > 0:
                    # 調整驚嚇時的得分策略
                    if scaredTime > 2:
                        score += 20 / (ghostDistance + 1e-5)  # 驚嚇時期鼓勵接近鬼魂
                    else:
                        score -= 10 * (safeDistance / (ghostDistance + 1e-5))  # 驚嚇即將結束，提高逃避的動機
                else:
                    score -= 10 * (safeDistance / (ghostDistance + 1e-5))  # 未驚嚇時嚴重懲罰近距離接觸
        
        return successorGameState.getScore() + score  # 結合計算的評分和遊戲的基本分數

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def maxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, 1))
            return v

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('inf')
            nextAgent = agentIndex + 1
            if nextAgent >= state.getNumAgents():
                nextAgent = 0  # Loop back to Pacman
                depth += 1     # Increase depth since all agents have moved
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if nextAgent == 0:
                    v = min(v, maxValue(successor, depth, nextAgent))
                else:
                    v = min(v, minValue(successor, depth, nextAgent))
            return v

        # Use maxValue to start Minimax computation
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):  # Pacman is always agent 0
            successor = gameState.generateSuccessor(0, action)
            score = minValue(successor, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alpha_beta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            if agentIndex == 0:  # Max agent (Pacman)
                return max_value(state, depth, agentIndex, alpha, beta)
            else:  # Min agent (ghosts)
                return min_value(state, depth, agentIndex, alpha, beta)
        
        def max_value(state, depth, agentIndex, alpha, beta):
            v = float('-inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alpha_beta(successor, depth + 1, (agentIndex + 1) % state.getNumAgents(), alpha, beta))
                if v > beta:  # 使用嚴格大於
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alpha_beta(successor, depth + 1, (agentIndex + 1) % state.getNumAgents(), alpha, beta))
                if v < alpha:  # 使用嚴格小於
                    return v
                beta = min(beta, v)
            return v

        # Choose the best action for Pacman
        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = alpha_beta(successor, 1, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            if agentIndex == 0:  # MAX agent (Pacman)
                return max_value(state, depth, agentIndex)
            else:  # EXP agent (ghosts)
                return exp_value(state, depth, agentIndex)
        
        def max_value(state, depth, agentIndex):
            v = float('-inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents()))
            return v
        
        def exp_value(state, depth, agentIndex):
            v = 0
            actions = state.getLegalActions(agentIndex)
            p = 1 / len(actions)  # Assuming uniform probability distribution
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v += p * expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents())
            return v

        # Choose the best action for Pacman
        bestAction = None
        bestScore = float('-inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. 分數影響的因子
    2. Food: 考慮最近食物的距離
    3. Ghost: 與 Ghost 的距離以及是否是可吃掉 Ghost 的狀態
    4. Capsule: 考慮 Capsule 的位置(但實測後發現考慮膠囊後分數較低故先拿掉)
    
    """
    
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # newCapsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    
    # 考慮食物距離
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        score += 1 / min(foodDistances)  # 越接近食物，分數越高

    # 考慮鬼魂的距離和恐懼狀態
    for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
        ghostDistance = manhattanDistance(newPos, ghost.getPosition())
        if scaredTime > 0:
            # 鬼魂害怕時，鼓勵接近鬼魂
            score += 10 / ghostDistance
        else:
            # 鬼魂不害怕時，避免接近鬼魂
            score -= 10 / ghostDistance if ghostDistance else -100  # 避免除以0的錯誤

    # # 考慮膠囊的位置
    # if newCapsules:
    #     capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
    #     score += 5 / min(capsuleDistances)  # 越接近膠囊，分數越高

    return score
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
