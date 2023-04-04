import jax.numpy as jnp
from jax.scipy.signal import convolve, boxcar_kernel

def fact_placement_score(img_feat):
    ice = img_feat[0] / 100
    ore = img_feat[1] / 100
    rubble = img_feat[2] / 100
    own_factories = img_feat[3]
    opp_factories = img_feat[9]
    ice_weight = 2.
    ore_weight = 1.
    rubble_weight = -0.05
    fact_weight = -0.5

    distances = jnp.zeros_like(ice)

    kernel_size = 3
    num_iterations = 3

    for i in range(num_iterations):
        # Blur the distances array using a boxcar kernel
        kernel = boxcar_kernel(kernel_size)
        blurred_distances = convolve(distances.reshape((1, -1, -1)), kernel.reshape((kernel_size, kernel_size)), mode="same")
        
        # Update the distances array using the blurred versions of each feature map
        distances = ice_weight * blurred_distances * ice + ore_weight * blurred_distances * ore + rubble_weight * blurred_distances * rubble + fact_weight * blurred_distances * own_factories + fact_weight * blurred_distances * opp_factories

    return distances


def bidding(img_feat):
    ice = img_feat[0] / 100
    ore = img_feat[1] / 100
    return {"bid" : int(20 * jnp.mean(ice) + jnp.mean(ore)), "faction" : "TheBuilders"}