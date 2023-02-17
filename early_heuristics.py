from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree

def manhattan_distance(binary_mask):
    distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
    return distance_map

def manhattan_dist_to_nth_closest(arr, n):
    if n == 1:
        distance_map = distance_transform_cdt(1-arr, metric='taxicab')
        return distance_map
    else:
        true_coords = np.transpose(np.nonzero(arr)) 
        tree = KDTree(true_coords) 
        dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n, p=1)
        return np.reshape(dist[:, n-1], arr.shape) 

def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1):
    
    def dfs(array, loc):
        distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
        if not (0<=loc[0]<array.shape[0] and 0<=loc[1]<array.shape[1]):   # check to see if we're still inside the map
            return 0
        if (not array[loc]) or visited[loc]:     # we're only interested in low rubble, not visited yet cells
            return 0
        if not (min_dist <= distance_from_start <= max_dist):      
            return 0

        visited[loc] = True

        count = 1.0 * exponent**distance_from_start
        count += dfs(array, (loc[0]-1, loc[1]))
        count += dfs(array, (loc[0]+1, loc[1]))
        count += dfs(array, (loc[0], loc[1]-1))
        count += dfs(array, (loc[0], loc[1]+1))

        return count

    visited = np.zeros_like(array, dtype=bool)
    return dfs(array, start)

def get_ice_and_ore(obs):
    ice = obs["player0"]["board"]["ice"]
    ore = obs["player_0"]["board"]["ore"]
    return {"ice" : ice, "ore" : ore}

def get_distances(distDict, numClosest):
    ice_dist = manhattan_dist_to_nth_closest(distDict["ice"], numClosest[0])
    ore_dist = manhattan_dist_to_nth_closest(distDict["ore"], numClosest[1]) 
    return {"ice" : ice_dist, "ore" : ore_dist}