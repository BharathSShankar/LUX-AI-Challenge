from typing import Tuple
import flax.linen as nn
import jax.numpy as jnp

class SELayer(nn.Module):
    """Squeeze-excitation layer for ResNet."""

    channels: int

    @nn.compact
    def __call__(self, x):
        x_global_avg = jnp.mean(x, axis=(1, 2), keepdims=True)
        x = nn.Dense(self.channels // 16, name='fc1')(x_global_avg)
        x = nn.relu(x)
        x = nn.Dense(self.channels, name='fc2')(x)
        x = nn.sigmoid(x)
        return x * x_global_avg

class ResNetBlock(nn.Module):
    """Residual block for ResNet."""

    channels: int
    strides: Tuple[int, int] = (1, 1)
    use_projection: bool = False
    use_se: bool = False
    ksize : int = 3
    @nn.compact
    def __call__(self, x):
        shortcut = x

        # Projection shortcut in case input and output shapes are different
        if self.use_projection:
            shortcut = nn.Conv(self.channels, self.strides, (1, 1), name='shortcut_conv')(shortcut)
            shortcut = nn.BatchNorm(name='shortcut_bn')(shortcut)

        x = nn.Conv(self.channels, (self.ksize, self.ksize), self.strides, name='conv1')(x)
        x = nn.BatchNorm(name='bn1')(x)
        x = nn.relu(x)

        x = nn.Conv(self.channels, (self.ksize, self.ksize), (1, 1), name='conv2')(x)
        x = nn.BatchNorm(name='bn2')(x)

        if self.use_se:
            x = SELayer(self.channels)(x)

        return nn.relu(x + shortcut)

