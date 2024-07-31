# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
from util import PriorityQueue
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Priority queue will store the nodes to be visited
    nodes = PriorityQueue()
    fringe = []
    # to store the path to reach present node
    path = []
    # easy to find visited node storing them in below set
    explored=set([])
    # initial cost is 0
    priority=0
    # dictionary will store cost to node
    dict={}
    # starting node where pacman will start
    start_node = problem.getStartState()
    if problem.isGoalState(start_node): # checking starting node and node where food is present
        return 'already in goal'
    else:
        # track visited node here store in queue
        nodes.push((start_node, path),priority)
        # cost to start is 0
        dict[start_node] = 0
        # add node to visited or explored
        explored.add(start_node)
        # continue till there is node in queue
        while (nodes):
            # pop node that is visited and its path
            curr, path = nodes.pop()
            # again check is there is food
            if problem.isGoalState(curr):
                return path
            else:
                # explore node successor nodes
                next = problem.getSuccessors(curr)
                for node in nodes.heap:
                    fringe.append(node[0]) # add to fringe or list
                for states in next:
                        # check if successor is there in dictionary
                        if states[0] not in (key for key in dict):
                            cost=problem.getCostOfActions(path + [states[1]])
                            nodes.push((states[0], path + [states[1]]),cost)
                            # update the cost of node visited
                            dict[states[0]]=cost
                            # add it to list of explored nodes
                            explored.add(states[0])
                        # incase it is there in dict but there is possible path of less cost
                        elif states[0] in (key for key in dict) and (problem.getCostOfActions(path + [states[1]]) < dict[states[0]]) :
                            cost = problem.getCostOfActions(path + [states[1]])
                            nodes.push((states[0], path + [states[1]]), cost)
                            # getting the minimum cost to visit that node
                            dict[states[0]] = cost
                            # add this to explored
                            explored.add(states[0])
    # in case there is no food for pacman
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    # pacman will begin from this node
    start_state = problem.getStartState()
    # Priority queue 
    frontier = PriorityQueue()
    # dictionary to store visited node
    parents_dictionary = {}
    actions = myAStar(problem, start_state, frontier, heuristic, parents_dictionary)
    return actions

def myAStar(problem, current_position, frontier, heuristic, parents_dictionary):
    visited = set()

    # Priority is the sum of the cost and heuristic value
    # initial set to 0
    priority = 0
    # initial state into queue
    frontier.push((current_position, [], 0), priority)
    # continue till there is node in queue 
    while not frontier.isEmpty():
        current_position, actions, cost = frontier.pop()
        # in case current node is already there
        if current_position in visited:
            continue
        # append visited node position
        visited.add(current_position)
    # if present node is place where food is present
        if problem.isGoalState(current_position):
            return actions
        # all successor of present node
        successors = problem.getSuccessors(current_position)
        # loop to each successor
        for successor, action, step_cost in successors:
            # create a new list of actions by adding the current action
            if successor not in visited:
                 # update the total cost
                new_actions = actions + [action]
                new_cost = cost + step_cost
                priority = new_cost + heuristic(successor, problem)
                # push the successor in priority queue
                frontier.push((successor, new_actions, new_cost), priority)
    # in case there is no food return empty list
    return []  # Return an emp



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch