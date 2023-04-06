import jax.numpy as jnp
import jax

x = jnp.array([1, -9, -jnp.inf, 2, 9, 2])
print(jax.random.categorical(jax.random.PRNGKey(9), x))
