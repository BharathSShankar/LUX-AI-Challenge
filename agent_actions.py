import numpy as np
import heapq
import math

from lux.kit import GameState
from lux.cargo import UnitCargo
from lux.config import EnvConfig
from lux.team import Team, FactionTypes
from lux.unit import Unit
from lux.factory import Factory

# Helper Functions
def search(start, dxdy, robot, gameState):

	class Node:
	
		def __init__(self, state, parent, path, path_cost):
			self.state = state # Coordinate of current position
			self.parent = parent # Previous Node
			self.path = path # Actions Taken
			self.path_cost = path_cost # Total cost of actions taken
		
		def __lt__(self, other):
			return self.path_cost < other.path_cost
		
		def get_state(self):
			return self.state
		
		def get_parent(self):
			return self.parent
			
		def get_path(self):
			return self.path
			
		def get_path_cost(self):
			return self.path_cost
		
	board = gameState.board
	move_deltas = np.array([])
	if dxdy[0] > 0:
		move_deltas += [[1, 0]]
	elif dxdy[0] < 0:
		move_deltas += [[-1, 0]]
	if dxdy[1] > 0:
		move_deltas += [[0, 1]]
	elif dxdy[0] < 0:
		move_deltas += [[0, -1]]
	action_moves = []
	frontier = []
	explored = set()
	gHatBest = dict()
	frontier.append( Node(start, None, [], 0) )
	while len(frontier) > 0:
		curr_node = heapq.heappop(frontier)
		if curr_node.get_state() in explored:
			if curr_node.get_path_cost() < gHatBest[ curr_node.get_state() ][0]:
				gHatBest[ curr_node.get_state() ] = [curr_node.get_path_cost(), curr_node.get_path()]
			else:
				continue
		else:
			gHatBest[ curr_node.get_state() ] = [curr_node.get_path_cost(), curr_node.get_path()]
			explored.add( curr_node.get_state() )
		if curr_node.get_state() == dxdy:
			return curr_node.get_path(), curr_node.get_path_cost()
		else:
			for action in move_deltas:
				new_state = action + curr_node.get_state()
				factory_there = board.factory_occupancy_map[new_state[0], new_state[1]]
				if new_state[0] < 0 or new_state[1] < 0 or new_state[1] >= len(board.rubble) or new_state[0] >= len(board.rubble[0]):
					continue
				elif factory_there not in gameState.teams[robot.agent_id].factory_strains and factory_there != -1:
					continue
				elif (abs(new_state[0]) - abs(dxdy[0]) > 0) or (abs(new_state[1])-abs(dxdy[1])> 0):
					continue
				else:
					rubble_at_target = board.rubble[new_state[0]][new_state[1]]
					path = curr_node.get_path() + action
					path_cost = curr_node.get_path_cost() + math.floor(robot.unit_cfg.MOVE_COST + robot.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)
					heapq.heappush(frontier, Node(new_state, curr_node, path, path_cost))			


def best_path(end, robot, gameState):
	start = robot.pos
	dx = start[0] - end[0]
	dy = start[1] - end[1]
	dxdy = [dx, dy]
	actions = search(start, dxdy, robot, gameState)
	return actions
		
# Intermediate Action Space
def harvester(res_coord: np.ndarray, robot:Unit, gameState: GameState):
	action_queue, path_cost = best_path(res_coord, robot, gameState)
	# TODO: Simplify action_queue movement into one action if possible
	# TODO: add sequence of movement for digging and transporting (rule-based?)
	robot.action_queue = action_queue
	pass

def expand(robot: Unit, gameState: GameState):
	"""
	Clear Rubble around Factory for Lichen to Grow
	"""
	# TODO: In the direction with less rubble, dig rubble near-to-far factory
	pass

def explorer(direction: np.ndarray, robot: Unit, gameState: GameState):
	"""
	Clear Rubble in one direction to open path
	"""
	# TODO: In the direction towards a position, dig rubble towards it
	pass

def energy_supplier(ally_target:Unit, robot: Unit, gameState: GameState):
	"""
	Supply Power to a particular unit
	"""
	pass

def resource_transporter(ally_target:Unit, robot: Unit, gameState: GameState)):
	"""
	Transport resource back to factory from a harvester
	"""
	pass

def attack_lichen(direction: np.ndarray, robot: Unit, gameState: GameState):
	"""
	Move to enemy lichen area and randomly dig
	"""
	pass

def attack_harvest(direction: np.ndarray, robot: Unit, gameState: GameState):
	"""
	Move to Enemy Resource area and randomly move around
	"""
	pass

# High Level Strategy/Actions
def economy():
	"""
	Focus on gathering resources only
	"""
	pass

def expansion():
	"""
	Focus on exploring and creating paths while maintaing resources at a safety level
	"""
	pass

def rush():
	"""
	Focus on attacking enemy while maintaing resources at a safety level
	"""
	pass

def full_rush():
	"""
	Move all units into enemy area without maintaining resources at safety level at longer
	"""
	pass