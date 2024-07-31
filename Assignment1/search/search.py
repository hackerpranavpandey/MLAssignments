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

# Creating Stack a container which follows LIFO property
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.isEmpty():
            return self.items.pop()

    def peek(self):
        if not self.isEmpty():
            return self.items[-1]

    def size(self):
        return len(self.items)


def depthFirstSearch(problem):
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
    #created a class for stack above and then a object frontier
    frontier = Stack()
    # each time problem will give initial state of the game
    start_state = problem.getStartState()
    # track of present node , total cost so far and its parent node
    node = {'state': start_state, 'action': None, 'cost': 0, 'parent': None}

    # push dictionary of first node details
    frontier.push(node)

    # now we need to maintain a set in order to keep track of visited nodes so far
    closed = set()

    # loop will run until the stack is empty
    while not frontier.isEmpty():
        # each time pop the top element in the stack
        node = frontier.pop()

        # so goal state is state where food is present if present state is that
        # here goal state is inbuilt function which return true if present state is goal state
        if problem.isGoalState(node['state']):
            actions = []
            # if present state is not null
            while node['parent'] is not None:
                # insert that state in action
                actions.insert(0, node['action'])  
                # changing node variable here
                node = node['parent']
            return actions

        # if present state is not visited
        if node['state'] not in closed:
            # insert it in set
            closed.add(node['state'])
            # getting all successor of present node
            successors = problem.getSuccessors(node['state'])
            # visiting all possible successors
            for successor in successors:
                # state action cost and parent of successor
                child = {
                    'state': successor[0],
                    'action': successor[1],
                    'cost': successor[2],
                    'parent': node
                }
                # push it in stack now
                frontier.push(child)

    # when there is no solution return empty list
    return []


# Creating Queue a container which follows FIFO property
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.isEmpty():
            return self.items.pop(0)

    def size(self):
        return len(self.items)


def breadthFirstSearch(problem):
   # above a class for queue data structure is created and then here a object for that class
    frontier = Queue()
    start_state = problem.getStartState()  # starting state for the problem 
    # total cost to visit present node
    node = {'state': start_state, 'action': None, 'cost': 0, 'parent': None}

    # queue data structure will visit level wise
    # insert present state in queue
    frontier.enqueue(node)

    # Keep track of visited states to avoid loops
    closed = set()

    while not frontier.isEmpty():
        # Dequeue the current state and actions from the queue
        node = frontier.dequeue()

        # Check if the current state is the goal state
        if problem.isGoalState(node['state']):
            actions = []
            while node['parent'] is not None:
                actions.insert(0, node['action'])  # Insert at the beginning to maintain order
                node = node['parent']
                # return the action if present state is goalState that we want to visit
            return actions

        # Check if the current state has been visited
        if node['state'] not in closed:
            # add the present state to set 
            closed.add(node['state'])
            # now we will find all successor of present state and visit them if not visited
            successors = problem.getSuccessors(node['state'])
            for successor in successors:
                child = {
                    'state': successor[0],
                    'action': successor[1],
                    'cost': successor[2],
                    'parent': node
                }
                # put them in queue
                frontier.enqueue(child)

    # If no solution is found, return an empty list
    return []






    

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
