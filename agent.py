from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
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
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]

        heavy_bot_tiles = []
        heavy_bot_units = []
        for unit_id, unit in units.items():
            if unit.unit_type == "HEAVY":
                heavy_bot_tiles += [unit.pos]
                heavy_bot_units += [unit]
        heavy_bot_tiles = np.array(heavy_bot_tiles)

        light_bot_tiles = []
        light_bot_units = []
        for unit_id, unit in units.items():
            if unit.unit_type == "LIGHT":
                light_bot_tiles += [unit.pos]
                light_bot_units += [unit]
        light_bot_tiles = np.array(light_bot_tiles)

        factory_tiles = []
        factory_units = []
        for unit_id, unit in factories.items():
            factory_tiles += [unit.pos]
            factory_units += [unit]
        factory_tiles = np.array(factory_tiles)

        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST and \
                    len(heavy_bot_units) == 0:
                actions[unit_id] = factory.build_heavy()
            if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST and \
                    len(light_bot_units) == 0 and len(heavy_bot_units) != 0 and not is_occupied(heavy_bot_tiles,light_bot_tiles,factory.pos):
                actions[unit_id] = factory.build_light()

            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()



        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)

        for index, items in enumerate(units.items()):
            unit_id, unit = items
            # print(unit.unit_type, file = sys.stderr)
            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            adjacent_to_light_bot = False
            adjacent_to_heavy_bot = False
            done_transfer = False
            if len(factory_tiles) > 0 and unit.unit_type == "HEAVY":
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                if len(light_bot_tiles) > 0:
                    light_bot_distances = np.mean((light_bot_tiles - unit.pos) ** 2, 1)
                    closest_light_bot_tile = light_bot_tiles[np.argmin(light_bot_distances)]
                    closest_light_bot = light_bot_units[np.argmin(light_bot_distances)]
                    adjacent_to_light_bot = np.mean((closest_light_bot_tile - unit.pos) ** 2) == 0.5

                # previous ice mining code
                ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                if unit.cargo.ice != 0 and adjacent_to_light_bot and closest_light_bot.cargo.ice == 0:
                    if unit.cargo.ice < 100:
                        actions[unit_id] = [unit.transfer(direction_to(unit.pos, closest_light_bot_tile), 0,
                                                          unit.cargo.ice,
                                                          repeat=0, n=1)]
                    else:
                        actions[unit_id] = [unit.transfer(direction_to(unit.pos, closest_light_bot_tile), 0,
                                                          100,
                                                          repeat=0, n=1)]
                if np.all(closest_ice_tile == unit.pos):
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                else:
                    if game_state.board.rubble[unit.pos[0]][unit.pos[1]] != 0:
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]

                if adjacent_to_factory and unit.power != 1500:
                    actions[unit_id] = [unit.pickup(4, 1500 - unit.power, repeat=0, n=1)]

            if len(factory_tiles) > 0 and unit.unit_type == "LIGHT":
                ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]

                heavy_bot_distances = np.mean((heavy_bot_tiles - unit.pos) ** 2, 1)
                closest_heavy_bot_tile = heavy_bot_tiles[np.argmin(heavy_bot_distances)]
                closest_heavy_bot = heavy_bot_units[np.argmin(heavy_bot_distances)]
                adjacent_to_heavy_bot = np.mean((closest_heavy_bot_tile - unit.pos) ** 2) == 0.5

                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                if len(heavy_bot_tiles) > 0:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    move_cost = unit.move_cost(game_state, direction)

                    if adjacent_to_factory and unit.power <= 140:
                        actions[unit_id] = [unit.pickup(4, 150-unit.power, repeat=0, n=1)]
                    elif (move_cost is not None) and (
                            unit.power > (move_cost + unit.action_queue_cost(game_state))+ 20) and (
                            not adjacent_to_heavy_bot) and unit.cargo.ice == 0:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]

                    elif (move_cost is not None) and (unit.power <= (
                            move_cost + unit.action_queue_cost(game_state) + 20 )) and not adjacent_to_factory:
                        direction = direction_to(unit.pos, closest_factory_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            for dire in get_possible_directions(direction):
                                next_pos = unit.pos + move_deltas[dire]
                                if game_state.board.rubble[next_pos[0]][next_pos[1]] == 0:
                                    actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                                    break


                    elif adjacent_to_heavy_bot and \
                            ((closest_heavy_bot.unit_id in actions.keys() and (
                            np.all(actions[closest_heavy_bot.unit_id][0] == unit.dig(repeat=0, n=1)))) or
                             closest_heavy_bot.unit_id not in actions.keys()):
                        actions[unit_id] = [unit.transfer(direction_to(unit.pos, closest_heavy_bot_tile), 4,
                                                          (unit.power - move_cost - unit.action_queue_cost(game_state) - 20),
                                                          repeat=0, n=1)]

                    elif adjacent_to_factory and unit.cargo.ice != 0 :
                        actions[unit_id] = [unit.transfer(direction_to(unit.pos, closest_factory_tile), 0,
                                                          unit.cargo.ice,
                                                          repeat=0, n=1)]


        return actions


def is_occupied(heavy_bot_tiles, light_bot_tiles, pos):
    for tiles in heavy_bot_tiles:
        if np.all(pos == tiles):
            return True
    for tiles in light_bot_tiles:
        if np.all(pos == tiles):
            return True
    return False

def get_possible_directions(directioin):
    if directioin == 1:
        return [1,2,4]
    if directioin == 2:
        return [2,1,3]
    if directioin == 3:
        return [3,2,4]
    if directioin == 4:
        return [4,1,3]
