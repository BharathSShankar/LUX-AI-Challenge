import numpy as np
import heapq
import math

from lux.kit import GameState
from lux.cargo import UnitCargo
from lux.config import EnvConfig
from lux.team import Team, FactionTypes
from lux.unit import Unit
from lux.factory import Factory
from lux.utils import direction_to

# Helper Functions
dir_mapper = {[0, 0]: 0,
	       [0, -1]: 1,
	       [1, 0]: 2,
	       [0, 1]: 3,
	       [-1, 0]: 4
	       }

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

def compact_movement(action_queue, robot:Unit):
	compact_aq = []
	consec_seq = 1
	for action in range(1, len(action_queue)):
		prev_action = action_queue[action-1]
		curr_action = action_queue[action]
		if curr_action == prev_action:
			consec_seq += 1
			continue
		else:
			compact_aq += [ robot.move(dir_mapper[prev_action], consec_seq, 1) ]
			consec_seq = 1
	compact_aq += [ robot.move(dir_mapper[curr_action], consec_seq, 1) ]
	return compact_aq

def reversed_movement(action_queue):
	reverse_queue = []
	for i in range(len(action_queue)):
		direction = action_queue[i][1]
		repeat = action_queue[i][4]
		if direction == 2: #Right
			reverse_queue += [ [0, 4, 0, 0, repeat, 1] ]
		elif direction == 4: #Left
			reverse_queue += [ [0, 2, 0, 0, repeat, 1] ]
		elif direction == 1: #Up
			reverse_queue += [ [0, 3, 0, 0, repeat, 1] ]
		else: #Down
			reverse_queue += [ [0, 1, 0, 0, repeat, 1] ]
	return reverse_queue

# Intermediate Action Space
def harvester(res_coord: np.ndarray, robot:Unit, gameState: GameState):
	action_queue, path_cost = best_path(res_coord, robot, gameState)
	compact_aq = compact_movement(action_queue, robot)
	# Compact the action_queue so that can fit more instructions behind
		
	while len(compact_aq) < 20:
		if robot.unit_type == "Light":
			remainingPowerAfterMove = robot.power - path_cost
			numberDigs = (remainingPowerAfterMove * 0.9) // robot.dig_cost()
			compact_aq += [ robot.dig(numberDigs, 1) ]
			compact_aq += [ robot.transfer(direction_to(res_coord, dir_mapper[compact_aq[-1][0]]), 0 if gameState.board.ice[res_coord[0]][res_coord[1]] == 1 else 1, robot.cargo.ice if gameState.board.ice[res_coord[0]][res_coord[1]] == 0 else robot.cargo.ore, 0, 1) ]

		else:
			remainingPowerAfterMove = robot.power - path_cost
			numberDigs = (remainingPowerAfterMove * 0.8) // robot.dig_cost()
			compact_aq += [ robot.dig(numberDigs, 1) ]
			compact_aq += [ robot.transfer(direction_to(res_coord, dir_mapper[compact_aq[-1][0]]), 0 if gameState.board.ice[res_coord[0]][res_coord[1]] == 1 else 1, robot.cargo.ice if gameState.board.ice[res_coord[0]][res_coord[1]] == 0 else robot.cargo.ore, 0, 1) ]
		if len(compact_aq) == 19:
			compact_aq += [ robot.move(dir_mapper[[0,0]], 50, 1)]

	robot.action_queue = compact_aq

def expander(robot: Unit, gameState: GameState):
	"""
	Clear Rubble around Factory for Lichen to Grow
	"""
	#robot.action_queue = action_queue

def explorer(direction: int, robot: Unit, gameState: GameState):
	"""
	Clear Rubble in one direction to open path
	"""
	# TODO: In the direction towards a position, dig rubble towards it
	#robot.action_queue = action_queue
	pass

def energy_supplier(ally_target:Unit, robot: Unit, gameState: GameState):
	"""
	Supply Power to a particular unit
	"""
	compact_aq = [ robot.pickup(4, robot.env_cfg.ROBOTS[robot.unit_type]["BATTERY_CAPCITY"] - robot.power) ]
	action_queue, path_cost = best_path(ally_target.pos, robot, gameState)
	compact_moves = compact_movement(action_queue, robot)
	compact_aq += [ compact_moves[:-1] ]
	compact_aq += [ robot.transfer(direction_to(ally_target.pos - action_queue[-1], ally_target.pos), 4, robot.env_cfg.ROBOTS[robot.unit_type]["BATTERY_CAPCITY"] - 2.5*path_cost, 0, 1) ]
	compact_aq += [ reversed_movement(compact_moves) ]
	robot.action_queue = compact_aq

def resource_transporter(ally_target:Unit, robot: Unit, gameState: GameState)):
	"""
	Transport resource back to factory from a harvester
	"""
	action_queue, path_cost = best_path(ally_target.pos, robot, gameState)
	compact_moves = compact_movement(action_queue, robot)
	compact_aq += [ compact_moves[:-1] ]
	compact_aq += [ robot.move(0, 10, 1) ]
	compact_aq += [ reversed_movement(compact_moves) ]
	robot.action_queue = compact_aq

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

# Higher Level Strategy which limit Actions Available
class Economy:
	
	"""
	Focus on Economy, Giving more rewards to harvesting resources
	"""

	def __init__(self):
		self.actions_available = [
				"harvester",
			    "energy_supplier",
				"resouce_transporter",
				"expander",
				"explorer"
			]

		# Rewards Change?
		#self.rewards_value = []

	def get_possible_actions(self):
		return self.actions_available

class Expansion:

	"""
	Focus on exploring and creating paths while maintaing resources at a safety level
	"""
	
	def __init__(self):
		self.actions_available = [
				"harvester",
			    "energy_supplier",
				"resouce_transporter",
				"expander",
				"explorer"
			]
		
		# Rewards Change?
		#self.rewards_value = []
		
	def get_possible_actions(self):
		return self.actions_available

class Rush:
	
	"""
	Focus on attacking enemy while maintaing resources at a safety level
	"""
	
	def __init__(self):
		self.actions_available = [
				"harvester",
				"explorer",
				"attack_lichen",
				"attack_harvest"
			]
		
		# Rewards Change?
		#self.rewards_value = []
		
	def get_possible_actions(self):
		return self.actions_available


class FullRush:
	
	"""
	Move all units into enemy area without maintaining resources at safety level at longer
	"""
	
	def __init__(self):
		self.actions_available = [
				"attack_lichen",
				"attack_harvest"
			]
		
		# Rewards Change?
		#self.rewards_value = []
		
	def get_possible_actions(self):
		return self.actions_available