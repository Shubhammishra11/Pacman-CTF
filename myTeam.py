from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint
import sys
sys.path.append("teams/<COMPAI>/")

# Create Team


def createTeam(firstIndex, secondIndex, isRed, first='AttackingAgent', second='DefendingAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class SmartActions():
    """
    A base class for actions, a defending or attacking can smartly choose.
    """

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

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.agent.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class getOffensiveActions(SmartActions):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        self.agent.distancer.getMazeDistances()
        self.counter = 0

        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2
        else:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Compute score from successor state
        features['successorScore'] = self.agent.getScore(successor)
        # get current position of the agent

        CurrentPosition = successor.getAgentState(self.index).getPosition()

        # Compute the distance to the nearest boundary
        boundaryMin = 10000000
        for i in range(len(self.boundary)):
            disBoundary = self.agent.getMazeDistance(
                CurrentPosition, self.boundary[i])
            if (disBoundary < boundaryMin):
                boundaryMin = disBoundary
        features['returned'] = boundaryMin

        features['carrying'] = successor.getAgentState(self.index).numCarrying
        # Compute distance to the nearest food
        foodList = self.agent.getFood(successor).asList()
        if len(foodList) > 0:
            minFoodDistance = 9999999
            for food in foodList:
                distance = self.agent.getMazeDistance(CurrentPosition, food)
                if (distance < minFoodDistance):
                    minFoodDistance = distance
            features['distanceToFood'] = minFoodDistance

        # Compute distance to the nearest capsule
        capsuleList = self.agent.getCapsules(successor)
        if len(capsuleList) > 0:
            minCapsuleDistance = 9999999
            for c in capsuleList:
                distance = self.agent.getMazeDistance(CurrentPosition, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
            features['distanceToCapsule'] = minCapsuleDistance
        else:
            features['distanceToCapsule'] = 0

        # Compute distance to closest ghost
        opponentsState = []
        for i in self.agent.getOpponents(successor):
            opponentsState.append(successor.getAgentState(i))
        visible = filter(
            lambda x: not x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(
                positions, key=lambda x: self.agent.getMazeDistance(CurrentPosition, x))
            closestDist = self.agent.getMazeDistance(CurrentPosition, closest)
            if closestDist <= 5:
                # print(CurrentPosition,closest,closestDist)
                features['GhostDistance'] = closestDist

        else:
            probDist = []
            for i in self.agent.getOpponents(successor):
                probDist.append(successor.getAgentDistances()[i])
                # print(probDist)
            features['GhostDistance'] = min(probDist)

        # Attacker only try to kill the enemy if : itself is ghost form and the distance between him and the ghost is less than 4
        enemiesPacMan = [successor.getAgentState(
            i) for i in self.agent.getOpponents(successor)]
        Range = filter(lambda x: x.isPacman and x.getPosition()
                       != None, enemiesPacMan)
        if len(Range) > 0:
            positions = [agent.getPosition() for agent in Range]
            closest = min(
                positions, key=lambda x: self.agent.getMazeDistance(CurrentPosition, x))
            closestDist = self.agent.getMazeDistance(CurrentPosition, closest)
            if closestDist < 4:
                # print(CurrentPosition,closest,closestDist)
                features['distanceToEnemiesPacMan'] = closestDist
        else:
            features['distanceToEnemiesPacMan'] = 0

        return features

    def getWeights(self, gameState, action):
        # If opponent is scared, the agent should not care about GhostDistance
        successor = self.getSuccessor(gameState, action)
        numOfCarrying = successor.getAgentState(self.index).numCarrying
        opponents = [successor.getAgentState(
            i) for i in self.agent.getOpponents(successor)]
        visible = filter(
            lambda x: not x.isPacman and x.getPosition() != None, opponents)
        if len(visible) > 0:
            for agent in visible:
                if agent.scaredTimer > 0:
                    if agent.scaredTimer > 12:
                        return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -1, 'distanceToCapsule': 0, 'returned': 10-3*numOfCarrying, 'carrying': 350}

                    elif 6 < agent.scaredTimer < 12:
                        return {'successorScore': 110+5*numOfCarrying, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -1, 'distanceToCapsule': -10, 'returned': -5-4*numOfCarrying,
                                'carrying': 100}

                # Visible and not scared
                else:
                    return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                            'GhostDistance': 20, 'distanceToCapsule': -15, 'returned': -15,
                            'carrying': 0}
        # Did not see anything
        self.counter += 1
        #print("Counter ",self.counter)
        return {'successorScore': 1000+numOfCarrying*3.5, 'distanceToFood': -7, 'GhostDistance': 0, 'distanceToEnemiesPacMan': 0,
                'distanceToCapsule': -5, 'returned': 5-numOfCarrying*3, 'carrying': 350}

    def allSimulation(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)

            reversed_direction = Directions.REVERSE[new_state.getAgentState(
                self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            a = random.choice(actions)
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(self.evaluate(next_state, Directions.STOP))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)
        current_direction = new_state.getAgentState(
            self.index).configuration.direction
        # The agent should not use the reverse direction during simulation

        reversed_direction = Directions.REVERSE[current_direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)

        # Randomly chooses a valid action
        for a in actions:
            # Compute new state and update depth
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(
                self.evaluate(next_state, Directions.STOP) + decay * self.allSimulation(depth - 1, next_state, decay))
        return max(result_list)

    def chooseAction(self, gameState):
        start = time.time()

        # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        feasible = []
        for a in actions:
            value = 0
            value = self.allSimulation(
                2, gameState.generateSuccessor(self.agent.index, a), 0.7)
            feasible.append(value)

        bestAction = max(feasible)
        possibleChoice = filter(
            lambda x: x[0] == bestAction, zip(feasible, actions))
        if((time.time() - start) > 1.5):
            print 'eval time for offensive agent %d: %.4f' % (
                self.agent.index, time.time() - start)
        return random.choice(possibleChoice)[1]


class getDefensiveActions(SmartActions):
    # Load the denfensive information
    def __init__(self, agent, index, gameState):
        self.index = index
        self.agent = agent
        self.DenfendList = {}

        if self.agent.red:
            middle = (gameState.data.layout.width - 2) / 2
        else:
            middle = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.boundary.append((middle, i))

        self.target = None
        self.lastObservedFood = None
        # Update probabilities to each patrol point.
        self.ProbableDefense(gameState)

    def ProbableDefense(self, gameState):
        """
        This method helps in choosing the target for defense. It calculates the
        distance from a position in boundary to defending foods and probability 
        is the inverse of the distance.
        By probability it decides that which food is most critical. So it targets
        to that food and tries to capture the opponent pacman.
        """
        total = 0

        for position in self.boundary:
            food = self.agent.getFoodYouAreDefending(gameState).asList()
            closestFoodDistance = min(
                self.agent.getMazeDistance(position, f) for f in food)
            if closestFoodDistance == 0:
                closestFoodDistance = 1
            self.DenfendList[position] = 1.0 / float(closestFoodDistance)
            total += self.DenfendList[position]

        # Normalize.
        if total == 0:
            total = 1
        for x in self.DenfendList.keys():
            self.DenfendList[x] = float(self.DenfendList[x]) / float(total)

    def selectPatrolTarget(self):
        """
        Select some patrol point to use as target.
        """

        maxProb = max(self.DenfendList[x] for x in self.DenfendList.keys())
        bestTarget = filter(
            lambda x: self.DenfendList[x] == maxProb, self.DenfendList.keys())
        return random.choice(bestTarget)

    def chooseAction(self, gameState):

        start = time.time()

        DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
        if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
            self.ProbableDefense(gameState)
        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None

        # Visible enemy , keep chasing.
        enemies = [gameState.getAgentState(
            i) for i in self.agent.getOpponents(gameState)]
        inRange = filter(
            lambda x: x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            eneDis, enemyPac = min(
                [(self.agent.getMazeDistance(myPos, x.getPosition()), x) for x in inRange])
            self.target = enemyPac.getPosition()

        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - \
                set(self.agent.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                closestFood, self.target = min(
                    [(self.agent.getMazeDistance(myPos, f), f) for f in eaten])

        self.lastObservedFood = self.agent.getFoodYouAreDefending(
            gameState).asList()

        # We have only a few dots.
        if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.agent.getFoodYouAreDefending(gameState).asList(
            ) + self.agent.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)

        # IF enough food is remaining than patrol at the main gate.
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        actions = gameState.getLegalActions(self.index)
        # feasible contains all actions that does lead to become stopped or where our bot is not a pacman
        feasible = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not a == Directions.STOP and not new_state.getAgentState(self.index).isPacman:
                newPosition = new_state.getAgentPosition(self.index)
                feasible.append(a)
                fvalues.append(self.agent.getMazeDistance(
                    newPosition, self.target))

        # best means minimum mazeDistance from our newposition to defending target
        best = min(fvalues)

        ties = filter(lambda x: x[0] == best, zip(fvalues, feasible))
        if((time.time() - start) >= 1.5):
            print 'eval time for teamPC1 defender agent %d: %.4f' % (
                self.index, time.time() - start)
        return random.choice(ties)[1]


class AttackingAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

    def chooseAction(self, gameState):
        if self.getScore(gameState) >= 13:
            return self.DefenceStatus.chooseAction(gameState)
        else:
            return self.OffenceStatus.chooseAction(gameState)


class DefendingAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

    def chooseAction(self, gameState):
        return self.DefenceStatus.chooseAction(gameState)
