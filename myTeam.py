# myTeam.py
# ---------
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


from sys import float_repr_style
from captureAgents import CaptureAgent
from util import manhattanDistance
from util import nearestPoint
import random, time, util, sys
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class DefensiveReflexAgent(CaptureAgent):
    """
    Defensive Agent that roams around base and looks for enemies
    """
    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.prev = []
      self.start = gameState.getAgentPosition(self.index)
      self.team = self.getTeam(gameState)
      self.goingToFood = None

    # getSuccessor from Reflex Agent
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
       
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
      """
      Returns the minimax action using 20 and evaluationFunction
      """
      "*** YOUR CODE HERE ***"
      import sys
      myState = gameState.getAgentState(self.index)
      myPos = myState.getPosition()
      food = self.getFood(gameState)
      foodMap = food.asList()
      numAgents = gameState.getNumAgents()
      #assign small to large negative number
      small = -sys.maxsize-1
      #assign large to large positive number
      large = sys.maxsize-1

      # modified Minimax function from multiagent
      def maxValue(gS, depth, agentIndex, a, b):
          #only takes in team agents
          if len(gS.getLegalActions(agentIndex)) == 0 or depth >= 20:
              return self.evaluationFunction(gS)
          v = -sys.maxsize-1
          if depth == 0:
                bestAction = gS.getLegalActions[0]

          for successor in gS.getLegalActions:
              nextBranch = gS.generateSuccessor(0, successor)
              nextBranchV = minValue(nextBranch, 1, depth+1, a, b)

              if nextBranchV > v:
                  v = nextBranchV
                  if depth == 0:
                      bestAction = successor
                
              if v > b:
                  return v
                    
              a = max(a, v)
            
          if depth == 0:
              return bestAction

          return v


      # modified Minimax function from multiagent
      def minValue(gS, depth, agent, a, b):
          if depth == 0 or depth >= 20 or agent in self.getTeam(gS):
              return self.evaluationFunction(gS)

          successorAgent = (agent + 1) % 4
          if successorAgent in self.getTeam(gS):
              newDepth = depth + 1
              v = sys.maxsize-1

              for successor in gS.getLegalActions(agent):
                if agent == gS.getNumAgents() - 1:
                    v = min(v, maxValue(gS.generateSuccessor(agent, successor), depth, a, b))
                else:
                    v = min(v, minValue(gS.generateSuccessor(agent, successor), agent+1, depth, a, b))

                if v < a:
                    return v
                
                b = min(b, v)
              return v
              minVal = min(maxValue(gS.generateSuccessor(agent, newAction), newDepth) for newAction in gS.getLegalActions(agent))

          else:
              newB = b
              v = sys.maxsize-1
              for successor in gS.getLegalActions(agent):
                if agent == gS.getNumAgents() - 1:
                    v = min(v, maxValue(gS.generateSuccessor(agent, successor), depth, a, b))
                else:
                    v = min(v, minValue(gS.generateSuccessor(agent, successor), agent+1, depth, a, b))

                if v < a:
                    return v
                
                b = min(b, v)
              return v

          minVal = min(minValue(gS.generateSuccessor(agent, newAction), depth, successorAgent) for newAction in gS.getLegalActions(agent))
          return minVal

      a = -sys.maxsize-1
      b = sys.maxsize-1
      v = -sys.maxsize-1
      for actions in gameState.getLegalActions(self.index):
          score = minValue(gameState.generateSuccessor(self.index, actions), 0, self.index+1, a, b)
          if score > v:
              v = score
              action = actions
          if v > b:
              return action
          if v > a:
              a = v

      if self.goingToFood != None:
          if self.getMazeDistance(myPos,self.goingToFood) <= 1:
              foodDefendingBools = self.getFoodYouAreDefending(gameState)
              foodMap = foodDefendingBools.asList()
              self.goingToFood = random.choice(foodMap)

      successor = self.getSuccessor(gameState, action)
      self.prev.append(action)
      if len(self.prev) >= 10:
          self.prev.pop()
      return action


      def min(num):
          largest = sys.maxsize-1
          for num in num:
              if num < largest:
                  largest = num
              if largest < self.small:
                  return v
              if largest < self.large:
                  self.large = largest
          return largest

      
    def evaluationFunction(self, gameState):
        import sys
        score = self.getScore(gameState)
        myState = gameState.getAgentState(self.index)
        position = myState.getPosition()
        foodMap = self.getFoodYouAreDefending(gameState).asList()

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
        # if enemy is found, kill them
        if len(invaders) > 0:
            minDist = sys.maxsize-1
            for invader in invaders:
                dist = self.getMazeDistance(position, invader.getPosition())
                if minDist > dist:
                    minDist = dist

            if minDist != sys.maxsize-1:
                score -= minDist*2
            # if distance is 0, eat invader
            if minDist == 0:
                score += 100000
            if invaders[-1].scaredTimer is not 0:
              score -= 100000

        else: #if no invaders, roam home base
            if self.goingToFood == None:
                self.goingToFood = random.choice(foodMap)

            score -= self.getMazeDistance(position,self.goingToFood)

            if position in self.prev[0:9]:
                score -= 1000

            else:
                score += 100
                self.prev.append(position)
                if len(self.prev) > 10:
                    self.prev.pop()
        return score



class OffensiveReflexAgent(CaptureAgent):

  def registerInitialState(self, gameState): # do initial observation of the game
    '''
    Initial observation of the whole game grid
    '''
    # to use maze distance from CaptureAgent
    CaptureAgent.registerInitialState(self, gameState)

    # get starting position
    self.startPos = gameState.getInitialAgentPosition(self.index)

    # get game grid observation
    self.legalPos = [p for p in gameState.getWalls().asList(False)]
    self.wallPos = [p for p in gameState.getWalls().asList(True)]

    # get agent indexes from each team
    self.team = self.getTeam(gameState)
    self.enemies = self.getOpponents(gameState)

    # state
    self.attack = True # set false when going back home
    self.power = False # set true when the agent ate a capsule
    self.scared = False # set true when the agent is scared at home

    # STATE 1 helper variables
    self.tryingEntry = None
    self.triedEntry = None
    self.lastEntry = None
    self.backOffPoint = None
    
    # set points where agent turns into either ghost (home) or pacman (away)
    self.entry = []
    self.enemyEntry = []
    if self.red:
      self.x_entry = (gameState.data.layout.width - 2)/2 # left side of the grid
      self.x_enemyEntry = ((gameState.data.layout.width - 2)/2) + 1 # right side of the grid
    else:
      self.x_enemyEntry = (gameState.data.layout.width - 2)/2 # left side of the grid
      self.x_entry = ((gameState.data.layout.width - 2)/2) + 1 # right side of the grid
    
    for y_entry in range (1, gameState.data.layout.height-1):
      if not gameState.hasWall(int(self.x_entry), y_entry):
        self.entry.append((int(self.x_entry), y_entry))
      if not gameState.hasWall(int(self.x_enemyEntry), y_entry):
        self.enemyEntry.append((int(self.x_enemyEntry), y_entry))
    
    # helper variables
    self.maxFoodCarriage = 10 # max food carriage, used in observeState()
    self.minFoodLeft = 2 # minimum food left before team can win
    self.maxGhostDist = 5 # max distance to an observable ghost, used in 
    self.minScaredTimer = 5 # minimum scared ghost timer to consider ghost as scared

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor  
  
  ############################
  # DECISION MAKING FUNCTION #
  ############################

  def chooseAction(self, gameState):
    '''
    OFFENSIVE AGENT STRATEGY

    Offensive agent has 3 different STATES:
    - STATE 1: HOME
    - STATE 2: AWAY
      - A: COLLECT FOOD/ATTACK
      - B: GO BACK HOME
      - C: POWER UP
    
    Steps of the strategy:
    - Observe CURRENT state. observeState()
    - Do an AI method depending on current state: 
      - If STATE == 1: (Agent still at home)
        - Find entry point
        - If agent reached entry point, check if there's a ghost blocking 
      - If STATE == 2: (Agent is away)
        - Use either Expectimax or Monte Carlo or Reinforcement Learning to determine the next action
        - Observe SUCCESSOR state. getFeatures() and getWeights() to get heuristic evaluation values
    '''

    start = time.time()

    self.observeState(gameState)

    currentState = gameState.getAgentState(self.index)
    
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    
    # STATE 1
    if not currentState.isPacman:

      values = [self.evaluate(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      return random.choice(bestActions)

    # STATE 2
    else: 
      
      # CURRENTLY ACTION IS BASED ON HEURISTICS
      values = [self.evaluate(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      return random.choice(bestActions)

      # TODO: IMPLEMENT EXPECTIMAX LATER
      # value, bestAction = self.getMax(gameState, depth=2) 
      

  #############################
  # CURRENT STATE OBSERVATION #
  #############################
  
  def observeState(self, gameState):
    '''
    Make observations of the current state and
    set the state variables based on the observations.
    '''

    # get previous state
    previousGameState = self.getPreviousObservation()
    previousState = None
    if previousGameState is not None:
      previousState = previousGameState.getAgentState(self.index)
    
    # get current state
    currentPos = gameState.getAgentPosition(self.index)
    currentState = gameState.getAgentState(self.index)
    
    # get food information
    foodPos = self.getFood(gameState).asList()
    foodLeft = len(foodPos)
    self.foodCarried = currentState.numCarrying

    # get enemy's state
    enemiesState = [gameState.getAgentState(enemy) for enemy in self.enemies]

    # get detectable ghost
    self.ghostState = [ghost for ghost in enemiesState if not ghost.isPacman and ghost.getPosition() != None]

    # STATE 1 HOME
    if not currentState.isPacman:
      
      self.attack = True

      # if pacman just got back home/died
      if previousState == None or previousState.isPacman:

        # find the closest food
        foodClosest = None
        foodDist = min([self.getMazeDistance(currentPos, food) for food in foodPos])
        for food in foodPos:
          if self.getMazeDistance(currentPos, food) == foodDist:
            foodClosest = food

        # find closest entry to the closest food
        entryDist = min([self.getMazeDistance(foodClosest, (x_entry, y_entry)) \
          for x_entry, y_entry in self.entry if (self.x_enemyEntry, y_entry) in self.enemyEntry])
        for x_entry, y_entry in self.entry:
          if self.getMazeDistance(foodClosest, (x_entry, y_entry)) == entryDist and (self.x_enemyEntry, y_entry) in self.enemyEntry:
            self.tryingEntry = (x_entry, y_entry)
        
        self.lastEntry = None
        self.backOffPoint = None

      if currentPos == self.tryingEntry:

        # if there is a ghost detected
        if self.ghostState:
          ghostDist = sys.maxsize+1
          ghostDist = min([self.getMazeDistance(currentPos, ghost.getPosition()) for ghost in self.ghostState]) 

          # if ghost distance is too close, go to back off point
          if ghostDist < self.maxGhostDist:
            
            self.lastEntry = self.tryingEntry

            # find back off point
            self.backOffPoint = self.findPoint(gameState, dist=self.maxGhostDist)

          # safe to cross
          else:
            self.tryingEntry = None
        
        # safe to cross
        else:
          self.tryingEntry = None

      elif self.backOffPoint:

        # if back off point is reached
        if currentPos == self.backOffPoint:

          # find the closest food
          foodClosest = None
          foodDist = min([self.getMazeDistance(currentPos, food) for food in foodPos])
          for food in foodPos:
            if self.getMazeDistance(currentPos, food) == foodDist:
              foodClosest = food
            
          # find new trying entry
          entryDist = min([self.getMazeDistance(foodClosest, (x_entry, y_entry)) for x_entry, y_entry in self.entry \
            if (self.x_enemyEntry, y_entry) in self.enemyEntry \
            and self.getMazeDistance((x_entry, y_entry), self.lastEntry) > self.maxGhostDist])
          for x_entry, y_entry in self.entry:
            if self.getMazeDistance(foodClosest, (x_entry, y_entry)) == entryDist \
              and (self.x_enemyEntry, y_entry) in self.enemyEntry \
              and self.getMazeDistance((x_entry, y_entry), self.lastEntry) > self.maxGhostDist:
              self.tryingEntry = (x_entry, y_entry)

          self.backOffPoint = None
        
 
    # STATE 2 AWAY
    else:

      if foodLeft < self.maxFoodCarriage:
        self.maxFoodCarriage = 5
      
      # if food left is less or equal to 2 or carried to many food
      if foodLeft <= self.minFoodLeft or self.foodCarried >= self.maxFoodCarriage:
        self.attack = False

      # if there's ghost
      elif self.ghostState:
        ghostDist = sys.maxsize+1
        ghostDist = min([self.getMazeDistance(currentPos, ghost.getPosition()) for ghost in self.ghostState]) 

        # if ghost distance is too close, go back home
        if ghostDist < self.maxGhostDist: 
          self.attack = False
        else:
          self.attack = True
      
      # check the distance to closest food and closest entry
      entryDist = min([self.getMazeDistance(currentPos, (x_enemyEntry, y_entry)) for x_enemyEntry, y_entry in self.enemyEntry \
        if (self.x_entry, y_entry) in self.entry])
      foodDist = min([self.getMazeDistance(currentPos, food) for food in foodPos])
      if foodDist > entryDist and entryDist < 3 and self.foodCarried > 0:
        self.attack = False

      # check if ghost is scared
      if min([enemy.scaredTimer for enemy in enemiesState]) < self.minScaredTimer:
        self.power = False

      elif min([enemy.scaredTimer for enemy in enemiesState]) > self.minScaredTimer:
        self.power = True
        self.attack = True


  ###########################################
  # SUCCESSOR STATE EVALUATION / HEURISTICS #
  ###########################################


  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    return features * weights
  
  def getFeatures(self, gameState, action):
    '''
    Returns a counter of features of a particular action

    Observation of next state:
    - number of food left
    - distance to food
    - food carried
    - enemy's true distance
    - distance to nearest entry point
    - capsule distance
    - is ghost scared?
    '''
    features = util.Counter()

    # get current state
    currentState = gameState.getAgentState(self.index)
    currentPos = gameState.getAgentPosition(self.index)

    # get successor state
    successorState = self.getSuccessor(gameState, action)
    successorPos = successorState.getAgentPosition(self.index)

    # STATE 1 HOME
    if not currentState.isPacman:
      
      if self.backOffPoint:
        features['pointDist'] = self.getMazeDistance(successorPos, self.backOffPoint)
      
      elif self.tryingEntry and not self.backOffPoint:
        features['pointDist'] = self.getMazeDistance(successorPos, self.tryingEntry)
      
      # safe to cross
      elif not self.tryingEntry and successorPos in self.enemyEntry:
        features['pointDist'] = 0
      elif not self.tryingEntry and successorPos not in self.enemyEntry:
        features['pointDist'] = 1
    
    # STATE 2 AWAY
    else:

      # get food information
      currentFoodLeft = self.getFood(gameState).asList()
      successorFoodLeft = self.getFood(successorState).asList()
      if len(successorFoodLeft) <= self.minFoodLeft:
        features['foodLeftNum'] = 0
      features['foodLeftNum'] = len(successorFoodLeft)

      # find minimum food distance for both current and successor state
      minCurrentFoodDist = min(self.getMazeDistance(currentPos, food) for food in currentFoodLeft)
      successorFoodDist = [self.getMazeDistance(successorPos, food) for food in successorFoodLeft]
      features['foodDist'] = min(successorFoodDist) if len(successorFoodDist) else 0 # get the nearest food

      # if pacman is going to eat food in the successor state, set food distance as 0
      if minCurrentFoodDist < min(successorFoodDist) and len(currentFoodLeft) > len(successorFoodLeft):
        features['foodDist'] = 0
      
      # find minimum ghost true distances (detectable ones)
      ghostDist = []
      for enemy in self.enemies:
        if not successorState.getAgentState(enemy).isPacman: # get ghosts only
          ghostPos = successorState.getAgentPosition(enemy)
          if ghostPos != None:
            ghostDist.append(self.getMazeDistance(ghostPos, successorPos))
      features['ghostDist'] = min(ghostDist) if len(ghostDist) else 0 # get the nearest ghost

      # if ghost is still far, ignore
      if features['ghostDist'] >= self.maxGhostDist: 
        features['ghostDist'] = 0
      
      # find minimum distance to entry points 
      entryDist = [self.getMazeDistance(successorPos, entry) for entry in self.entry]
      features['entryDist'] = min(entryDist) # get the nearest entry point

      # find minimum distance to the capsules
      capsuleDist = [self.getMazeDistance(successorPos, capsulePos) for capsulePos in self.getCapsules(gameState)]
      if features['ghostDist'] > 0: # if being chased, look for capsule
        features['capsuleDist'] = min(capsuleDist) if len(capsuleDist) else 0 # get the nearest capsule

    return features

  def getWeights(self, gameState, action):
    '''
    Return weights of a particular action
    '''

    weights = util.Counter()

    # get current state
    currentState = gameState.getAgentState(self.index)
    currentPos = gameState.getAgentPosition(self.index)

    #################
    # STATE 1: HOME #
    #################

    if not currentState.isPacman:
      weights['pointDist'] = -1
      return weights
    
    #################
    # STATE 2: AWAY #
    #################

    else:

      # STATE 2A: COLLECT FOOD/ATTACK
      if self.attack and not self.power:
        weights['foodLeftNum'] = -100
        weights['foodDist'] = -10 # look for the shortest one
        weights['capsuleDist'] = 5 # try to avoid capsule

      # STATE 2B: GO BACK HOME
      elif not self.attack and not self.power:
        weights['capsuleDist'] = -15 # look for capsule
        weights['ghostDist'] = 12 # avoid ghost
        weights['entryDist'] = -10 
      
      # STATE 2C: POWER UP
      elif self.attack and self.power:
        weights['foodLeftNum'] = -10 # look for food
        weights['foodDist'] = -1
        weights['ghostDist'] = -10 # chase ghost if it's detectable!
      
      return weights

  #################
  # HELPER METHODS
  #################

  def findPoint(self, gameState, dist):
    '''
    Finding a point with a specified distance using BFS
    '''
    from util import Queue

    # helper variables
    queue = Queue() # data structure
    visited = [] # all of the nodes visited by the algorithm
    
    # push initial state to stack
    queue.push(gameState) 
 
    # iterate
    while (True):
        # if queue empty: return an empty array
        if queue.isEmpty():
            return []
        
        # else: dequeue/pop a node based on search strategy
        currentState = queue.pop()
        currentPos = currentState.getAgentPosition(self.index)
        
        # check if current node is already visited, to avoid double expansion
        if currentPos not in visited:
        # mark current node as visited
            visited.append(currentPos)
            
            # if current node == goal state: return direction
            new_x, new_y = currentPos
            if self.red:
              if new_x == self.x_entry - dist:
                return currentPos
            else:
              if new_x == self.x_entry + dist:
                return currentPos
            
            # expand children
            actions = currentState.getLegalActions(self.index)
            nextStates = [self.getSuccessor(currentState, action) for action in actions]
            
            for nextState in nextStates:
                # check if node has not been visited before
                nextPos = nextState.getAgentPosition(self.index)
                next_x, next_y = nextPos
                if nextPos not in visited:
                  if self.red:
                    if next_x <= self.x_entry:
                      queue.push(nextState)
                  else:
                    if next_x >= self.x_entry:
                      queue.push(nextState)