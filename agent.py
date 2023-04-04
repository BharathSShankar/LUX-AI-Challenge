from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
from numpy.lib.stride_tricks import as_strided
import sys
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    # TODO: Adjust Weight properly
    def bid_policy(self, obs, w1=5, w2=5, w3=0.1) -> int:
        ice_map = obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = obs["board"]["ore"]
        ore_tile_locations = np.argwhere(ore_map == 1)
        rubble_map = obs["board"]["rubble"]
        res_count = len(ice_tile_locations) + len(ore_tile_locations)
        if res_count < 6:
            w1 *= 2
        elif res_count < 12:
            w1 *= 1.5
        elif res_count > 18:
            w1 = 0
        else:
            res_count = 0
        ice_tile_distances = np.zeros((EnvConfig.map_size, EnvConfig.map_size))
        ore_tile_distances = np.zeros((EnvConfig.map_size, EnvConfig.map_size))
        for i in range(len(EnvConfig.map_size)):
            for j in range(len(EnvConfig.map_size)):
                ice_tile_distances[i, j] = np.mean((ice_tile_locations-[i,j])**2, 1)
                ore_tile_distances[i, j] = np.mean((ore_tile_locations-[i,j])**2, 1)
        combine_distances = np.add(ice_tile_distances, ore_tile_distances)
        smallest_distance = np.argmin(combine_distances)
        if smallest_distance < 12:
            w2*= 2
        elif smallest_distance < 24:
            w2*=1.5
        elif smallest_distance > 36:
            w2 = 0

        bid = w1 * res_count + w2 * smallest_distance
        return bid
    
    # TODO: Adjust Weight Properly
    def placement_policy(self, potential_spawns: np.ndarray, gameState: GameState):

        closestIceWeight = 0.5
        closestOreWeight = 0.1
        iceWeight = 0.05
        oreWeight = 0.01
        rubbleWeight = 0.001
        enemyFactoryWeight = -0.01

        # Distance to Ice Res Tiles
        ice_map = gameState.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ice_tile_distances = np.full((EnvConfig.map_size, EnvConfig.map_size), np.inf)

        # Distance to Ore Res Tiles
        ore_map = gameState.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)
        ore_tile_distances = np.full((EnvConfig.map_size, EnvConfig.map_size), np.inf)

        # Rubble Count 
        rubble_map = gameState.board.rubble
        rubble_count = np.full((EnvConfig.map_size, EnvConfig.map_size), np.inf)

        # Enemy Factory
        enemy_factory = gameState.factories["player_0" if self.player == "player_1" else "player_1"]
        enemy_factory_distances = np.zeros((EnvConfig.map_size, EnvConfig.map_size))

        for i in range(len(EnvConfig.map_size)):
            for j in range(len(EnvConfig.map_size)):
                if potential_spawns[i][j] == True:
                    # General Res Distances
                    ice_tile_distances[i, j] = iceWeight*np.mean((ice_tile_locations-[i,j])**2, 1)
                    ore_tile_distances[i, j] = oreWeight*np.mean((ore_tile_locations-[i,j])**2, 1)
                    
                    # Distance to Closest Ice Tile
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances[i, j])]
                    ice_tile_distances[i, j] += closestIceWeight*(np.linalg.norm(closest_ice_tile-[i, j]))

                    # Distance to Closest Ore Tile
                    closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances[i, j])]
                    ore_tile_distances[i, j] += closestOreWeight*(np.linalg.norm(closest_ore_tile-[i, j]))

                    # Rubble Mean Low Count (7, 7) window size - (3, 3) factory size
                    rubble_tile_count = 0
                    total_rubble_count = 0
                    for window_row in range(i-3,i+4): # (7, 7) window size
                        for window_col in range(j-3, j+4):
                            if window_row < 0 or window_row > EnvConfig.map_size or window_col < 0 or window_col > EnvConfig.map_size:
                                continue
                            else:
                                rubble_tile_count+=1
                                total_rubble_count += rubble_map[window_row][window_col]
                    for window_row in range(i-1, i+2): # (3, 3) factory size
                        for window_col in range(j-1, j+2):
                            if window_row < 0 or window_row > EnvConfig.map_size or window_col < 0 or window_col > EnvConfig.map_size:
                                continue
                            else:
                                rubble_tile_count-=1
                                total_rubble_count -= rubble_map[window_row][window_col]
                    rubble_count[i][j] = rubbleWeight * (total_rubble_count/rubble_tile_count)

                    # Enemy Factory Distance
                    if len(enemy_factory) > 0:
                        for unit_id, factory in enemy_factory.items():
                            enemy_factory_distances[i][j] += (enemyFactoryWeight*np.linalg.norm(factory.pos - [i, j]))

        combine_value = np.add(ice_tile_distances, ore_tile_distances, rubble_count, enemy_factory_distances)

        water_left = gameState.teams[self.player].water
        metal_left = gameState.teams[self.player].metal
        spawn_loc = potential_spawns[np.argmin(combine_value)]
        if water_left > 150 and metal_left > 150:
            metal_amount,water_amount = 150, 150
        else:
            metal_amount,water_amount = water_left, metal_left        
        return spawn_loc, metal_amount, water_amount

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            bid = self.bid_policy(obs)
            return dict(faction="AlphaStrike", bid=bid)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc, metal_amount, water_amount = Agent.placement_policy(potential_spawns, game_state)
                return dict(spawn=spawn_loc, metal=metal_amount, water=water_amount)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if self.env_cfg.max_episode_length - game_state.real_env_steps < 50:
                if factory.water_cost(game_state) <= factory.cargo.water:
                    actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0)]
        return actions
