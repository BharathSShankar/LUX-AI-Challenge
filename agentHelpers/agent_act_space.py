from gym import spaces
import jax.numpy as jnp

unit_action_space = spaces.Dict({
    "move": spaces.Dict({
        "direction": spaces.Discrete(5),
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
    "transfer": spaces.Dict({
        "direction": spaces.Discrete(5),
        "rType": spaces.Discrete(5),
        "amount": spaces.Box(0, 1, shape=(1,), dtype=jnp.float32),
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
    "pickup": spaces.Dict({
        "rType": spaces.Discrete(5),
        "amount": spaces.Box(0, 1, shape=(1,), dtype=jnp.float32),
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
    "dig": spaces.Dict({
        "direction": spaces.Discrete(5),
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
    "self-destruct": spaces.Dict({
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
    "recharge": spaces.Dict({
        "amount": spaces.Box(0, 1, shape=(1,), dtype=jnp.float32),
        "repeat": spaces.Discrete(9999),
        "n": spaces.Discrete(1, 9999),
    }),
})

fact_action_space = spaces.Discrete(3)