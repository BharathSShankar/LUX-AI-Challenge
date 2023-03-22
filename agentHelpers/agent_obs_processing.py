import jax.numpy as jnp

from agentHelpers.agent_constants import FACTORY_VEC, GIV_SIZE, IMG_FEATURES_SIZE, UNIT_VEC
from lux.kit import GameState
from lux.cargo import UnitCargo
from lux.config import EnvConfig
from lux.team import Team, FactionTypes
from lux.unit import Unit
from lux.factory import Factory

def global_2_vec(gameState: GameState, player: str) -> jnp.array:
    player = player
    opposition = "player_1" if player == "player_0" else "player_0"
    giv_size = GIV_SIZE
    stateSpace = jnp.zeros(giv_size).astype(jnp.float32)
    
    own_st_id = set()
    opp_st_id = set()

    stateSpace[0] = gameState.real_env_steps() // gameState.env_cfg.CYCLE_LENGTH
    stateSpace[1] = gameState.real_env_steps() % gameState.env_cfg.CYCLE_LENGTH
    stateSpace[2] = 0 if gameState.is_day() else 1 

    for factory in gameState.factories[player].values():
        stateSpace[5] += 1
        own_st_id.add(factory.strain_id)

        stateSpace[9] += factory.cargo.metal
        stateSpace[10] += factory.cargo.water
        stateSpace[11] += factory.cargo.ore
        stateSpace[12] += factory.cargo.ice
        stateSpace[13] += factory.power

    for factory in gameState.factories[opposition].values():
        stateSpace[19] += 1
        opp_st_id.add(factory.strain_id)
        
        stateSpace[23] += factory.cargo.metal
        stateSpace[24] += factory.cargo.water
        stateSpace[25] += factory.cargo.ore
        stateSpace[26] += factory.cargo.ice
        stateSpace[27] += factory.power

    for unit in gameState.units[player].values():
        if unit.unit_type == "LIGHT":
            stateSpace[7] += 1
        else:
            stateSpace[8] += 1
        stateSpace[14] += unit.cargo.metal 
        stateSpace[15] += unit.cargo.water 
        stateSpace[16] += unit.cargo.ore 
        stateSpace[17] += unit.cargo.ice 
        stateSpace[18] += unit.power 
    
    for unit in gameState.units[opposition].values():
        if unit.unit_type == "LIGHT":
            stateSpace[21] += 1
        else:
            stateSpace[22] += 1
        stateSpace[28] += unit.cargo.metal
        stateSpace[29] += unit.cargo.water
        stateSpace[30] += unit.cargo.ore
        stateSpace[31] += unit.cargo.ice
        stateSpace[32] += unit.power

    for i in range(gameState.env_cfg.map_size):
        for j in range(gameState.env_cfg.map_size):
            stateSpace[3] += gameState.board.ice[i][j]
            stateSpace[4] += gameState.board.ore[i][j] 

            if gameState.board.lichen_strains[i][j] in own_st_id:
                stateSpace[6] += gameState.board.lichen[i][j]
            if gameState.board.lichen_strains[i][j] in opp_st_id:
                stateSpace[20] += gameState.board.lichen[i][j]

    return stateSpace

def img_2_vec(gameState:GameState, player: str) -> jnp.array:
    player = player
    opposition = "player_1" if player == "player_0" else "player_0"
    img_size = IMG_FEATURES_SIZE
    stateSpace = jnp.zeros(img_size, gameState.env_cfg.map_size, gameState.env_cfg.map_size).astype(jnp.float32) 

    own_lic_str = set()
    opp_lic_str = set()

    for factory in gameState.factories[player].values():
        x, y = factory.pos
        stateSpace[0, x, y] = 1
        stateSpace[1, x, y] = factory.power
        stateSpace[2, x, y] = factory.cargo.ice
        stateSpace[3, x, y] = factory.cargo.ore
        stateSpace[4, x, y] = factory.cargo.metal
        stateSpace[5, x, y] = factory.cargo.water
        own_lic_str.add(factory.strain_id)

    for factory in gameState.factories[opposition].values():
        x,y = factory.pos
        stateSpace[6, x, y] = 1
        stateSpace[7, x, y] = factory.power
        stateSpace[8, x, y] = factory.cargo.ice
        stateSpace[9, x, y] = factory.cargo.ore
        stateSpace[10, x, y] = factory.cargo.metal
        stateSpace[11, x, y] = factory.cargo.water
        opp_lic_str.add(factory.strain_id)
    
    for unit in gameState.units[player].values():
        x, y = unit.pos
        if unit.unit_type == "LIGHT":
            stateSpace[12, x, y] = 1
        else:
            stateSpace[13, x, y] = 1
        stateSpace[14, x, y] = unit.power
        stateSpace[15, x, y] = unit.cargo.ice
        stateSpace[16, x, y] = unit.cargo.ore
        stateSpace[17, x, y] = unit.cargo.metal
        stateSpace[18, x, y] = unit.cargo.water
    
    for unit in gameState.units[opposition].values():
        x, y = unit.pos
        if unit.unit_type == "LIGHT":
            stateSpace[19, x, y] = 1
        else:
            stateSpace[20, x, y] = 1
        stateSpace[21, x, y] = unit.power
        stateSpace[22, x, y] = unit.cargo.ice
        stateSpace[23, x, y] = unit.cargo.ore
        stateSpace[24, x, y] = unit.cargo.metal
        stateSpace[25, x, y] = unit.cargo.water
    
    stateSpace[26, :, :] = gameState.board.ice
    stateSpace[27, :, :] = gameState.board.ore
    stateSpace[28, :, :] = gameState.board.rubble

    for x in range(gameState.env_cfg.map_size):
        for y in range(gameState.env_cfg.map_size):
            if gameState.board.lichen_strains[x][y] in own_lic_str:
                stateSpace[29, x, y] = gameState.board.lichen[x, y]
            if gameState.board.lichen_strains[x][y] in opp_lic_str:
                stateSpace[30, x, y] = gameState.board.lichen[x, y]
    
    return stateSpace

def fact_2_vec(gameState: GameState, player : str, factoryId: str) -> jnp.array:
    factory = gameState.factories[player][factoryId]
    fact_dim = FACTORY_VEC
    fact_vec = jnp.zeros((fact_dim))
    x, y = factory.pos
    
    fact_vec[0] = x
    fact_vec[1] = y

    fact_vec[2] = factory.power
    fact_vec[3] = factory.cargo.ice
    fact_vec[4] = factory.cargo.ore
    fact_vec[5] = factory.cargo.water
    fact_vec[6] = factory.cargo.metal

    lich_str = factory.strain_id
    lich_mask = gameState.board.lichen_strains == lich_str
    fact_vec[8] += jnp.sum(gameState.board.lichen * lich_mask)

    return fact_vec

def unit_2_vec(gameState: GameState, player: str, unitId: str) -> jnp.array:
    unit = gameState.units[player][unitId]
    unit_dim = UNIT_VEC
    unit_vec = jnp.zeros((unit_dim))
    x, y = unit.pos

    unit_vec[0] = x
    unit_vec[1] = y

    unit_vec[2] = unit.power
    unit_vec[3] = unit.cargo.ice
    unit_vec[4] = unit.cargo.ore
    unit_vec[5] = unit.cargo.water
    unit_vec[6] = unit.cargo.metal

    return unit_vec