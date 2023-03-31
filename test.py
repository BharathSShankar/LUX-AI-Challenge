import jax
import jax.numpy as jnp

actions = jnp.array([[1., 2., 4.], [1., 2., 4.]]) + jax.random.normal(jax.random.PRNGKey(0), shape = (2, 3)) * jnp.array([[1., 0.2, .8], [1., 0.2, .8]])
print(actions)
