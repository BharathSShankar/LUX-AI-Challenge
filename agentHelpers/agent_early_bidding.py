import numpy as np
from scipy.ndimage.filters import gaussian_filter

def fact_placement_score(img_feat: np.array):

    ice = gaussian_filter(img_feat[0] / 100, sigma = 7)
    ore = gaussian_filter(img_feat[1] / 100, sigma = 7)

    rubble = gaussian_filter(img_feat[2] / 100, sigma = 7)

    ice_weight = 2
    ore_weight = 1

    rubble_weight = - 0.5
    
    own_factories = gaussian_filter(img_feat[3], sigma = 7) 
    opp_factories = gaussian_filter(img_feat[9], sigma = 7) 

    fact_weight = - 0.5

    return ice * ice_weight + ore * ore_weight + rubble * rubble_weight + own_factories * fact_weight + opp_factories * fact_weight

def bidding():
    return dict(faction="AlphaStrike", bid=0)