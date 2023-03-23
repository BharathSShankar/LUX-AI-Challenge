from typing import List, Tuple
import flax.linen as nn

class FullNet(nn.Module):
    num_resid_layers:int
    channel_nums: List[int]
    k_sizes: List[int]
    hiddenDimUnit: int
    strides: Tuple[int, int] = (1, 1)
    