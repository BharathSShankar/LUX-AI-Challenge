from functools import partial
import jax
import jax.numpy as jnp
import gym
import numpy as np

fun = jax.grad(lambda x, y: x**2 + y / 2, argnums=(0, 1))
print(fun(1.0, 2.0))