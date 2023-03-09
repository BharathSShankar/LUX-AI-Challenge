import numpy as np

from agent_constants import CHANNEL_SIZE
from lux.kit import GameState
from lux.cargo import UnitCargo
from lux.config import EnvConfig
from lux.team import Team, FactionTypes
from lux.unit import Unit
from lux.factory import Factory

def complete_state_view(gameState: GameState, player:str) -> np.ndarray:
    player = player
    opposition = "player_1" if player == "player_0" else "player_0"
    channelSize = CHANNEL_SIZE
    stateSpace = np.zeros((channelSize, gameState.env_cfg.map_size, gameState.env_cfg.map_size)).astype(np.float32)
    
    # Ally Player
    ## channel[0]: Number of units
    ## channel[1]: Robot Type
    ## channel[2:5]: Resources (Power, Ice, Ore) of each unit
    ## channel[5]: Action Queue
    for unit in gameState.units[player].values():
        x, y = unit.pos
        stateSpace[0, x, y] += 1
        stateSpace[1, x, y] = unit.unit_type
        stateSpace[2, x, y] = unit.power
        stateSpace[3, x, y] = unit.cargo.ice
        stateSpace[4, x, y] = unit.cargo.ore
        stateSpace[5, x, y] = unit.action_queue

    ## channel[6]: Number of factories
    ## channel[7:12]: Resources (Power, Ice, Ore, Water, Metal) of each factory
    fplayer_fstrain_fpos_map = {"player_0":{}, "player_1":{}}
    for factory in gameState.factories[player].values():
        x, y = factory.pos
        stateSpace[6, x, y] += 1
        stateSpace[7, x, y] = factory.power
        stateSpace[8, x, y] = factory.cargo.ice
        stateSpace[9, x, y] = factory.cargo.ore
        stateSpace[10, x, y] = factory.cargo.water
        stateSpace[11, x, y] = factory.cargo.metal
        # To be used later for channel 12, 13, 14
        fplayer_fstrain_fpos_map[player][factory.strain_id] = [x, y]

    ## channel[12, 13, 14] factory_lichen_tile, factory_lichen, ally_total_lichen will be added at the end under map obs

    # Opposition
    ## channel[15]: Number of units
    ## channel[16]: Robot Type
    ## channel[17:20]: Resources (Power, Ice, Ore) of each unit
    ## channel[20]: Action Queue
    for unit in gameState.units[opposition].values():
        x, y = unit.pos
        stateSpace[15, x, y] += 1
        stateSpace[16, x, y] = unit.unit_type
        stateSpace[17, x, y] = unit.power
        stateSpace[18, x, y] = unit.cargo.ice
        stateSpace[19, x, y] = unit.cargo.ore
        stateSpace[20, x, y] = unit.action_queue

    ## channel[21]: Number of factories
    ## channel[22:27]: Resources (Power, Ice, Ore, Water, Metal) of each factory
    for factory in gameState.factories[player].values():
        x, y = factory.pos
        stateSpace[21, x, y] += 1
        stateSpace[22, x, y] = factory.power
        stateSpace[23, x, y] = factory.cargo.ice
        stateSpace[24, x, y] = factory.cargo.ore
        stateSpace[25, x, y] = factory.cargo.water
        stateSpace[26, x, y] = factory.cargo.metal

        # To be used later for channel 27, 28, 29
        fplayer_fstrain_fpos_map[opposition][factory.strain_id] = [x, y]

    ## channel[27, 28, 29] factory_lichen_tile, factory_lichen, oppo_total_lichen will be added at the end under map obs

    # Map Observation
    ## Channel[12:15, 27:30] Lichen mentioned above
    for width in range(len(gameState.env_cfg["map_size"])):
        for height in range(len(gameState.env_cfg["map_size"])):
            strain_id = gameState.board.lichen_strains[width][height]
            if strain_id == -1:
                continue
            else:
                if strain_id in fplayer_fstrain_fpos_map[player].keys():
                    f_xpos, f_ypos = fplayer_fstrain_fpos_map[player][strain_id]
                    stateSpace[12, f_xpos, f_ypos] += 1
                    stateSpace[13, f_xpos, f_ypos] += gameState.board.lichen[width][height]
                    stateSpace[14, f_xpos, f_ypos] += gameState.board.lichen[width][height]
                else:
                    f_xpos, f_ypos = fplayer_fstrain_fpos_map[opposition][strain_id]
                    stateSpace[27, f_xpos, f_ypos] += 1
                    stateSpace[28, f_xpos, f_ypos] += gameState.board.lichen[width][height]
                    stateSpace[29, f_xpos, f_ypos] += gameState.board.lichen[width][height]
                    
    ## channel[30]: Amount of rubbles
    stateSpace[30] = gameState.board.rubble
    ## channel[31:33]: Location of Ice, Ore
    stateSpace[31], stateSpace[32] = gameState.board.ice, gameState.board.ore
    ## channel[33]: N_TURNS_LEFT_TO_END_GAME
    stateSpace[33] = gameState.env_cfg.max_episode_length - gameState.real_env_steps()
    ## channel[34]: N_TURNS_LEFT_TO_END_CYCLE
    stateSpace[34] = gameState.env_cfg.CYCLE_LENGTH - (gameState.real_env_steps() % gameState.env_cfg.CYCLE_LENGTH)
    ## channel[35]: DAY_OR_NIGHT
    stateSpace[35] = 0 if gameState.is_day() else 1
    
    return stateSpace



