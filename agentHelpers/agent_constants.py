from enum import IntEnum
import jax.numpy as jnp

MAP_SIZE = 48
ROBOT_ACTIONS = 6
FACTORY_ACTIONS = 3
EARLY_VEC = 14
GIV_SIZE = 33
IMG_FEATURES_SIZE = 31
FACTORY_VEC = 8
UNIT_VEC = 7

class RobotAction(IntEnum):	
	MOVE = 0
	TRANSFER = 1
	PICKUP = 2
	DIG = 3
	SELF_DESTRUCT = 4
	RECHARGE = 5

class FactoryAction(IntEnum):
	BUILD_LIGHT = 0
	BUILD_HEAVY = 1
	WATER = 2

LIC_WT = 0.1
REW_WTS = jnp.array([
    50,
    0.005,
    0.006,
    0.003,
    0.004,
    0.0001,
    -50,
    -0.005,
    -0.006,
    -0.003,
    -0.004,
    -0.0001,
    1,
    10,
    -0.0001,
    -0.0001,
    0.001,
    0.002,
    0.0005,
    -1,
    -10,
    0.0001,
    0.0001,
    -0.001,
    -0.002,
    -0.0005,
])