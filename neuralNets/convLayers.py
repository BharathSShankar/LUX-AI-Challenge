import jax.numpy as jnp
import flax.linen as nn

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
    strides: tuple[int, int] = (1, 1)
    use_projection: bool = True
    use_se: bool = False
    ksize: int = 3

    @nn.compact
    def __call__(self, x):
        shortcut = x
        # Projection shortcut in case input and output shapes are different
        if self.use_projection:
            shortcut = nn.Conv(self.channels, kernel_size=(1, 1), strides=self.strides, name='shortcut_conv')(shortcut)
            shortcut = nn.LayerNorm(name='shortcut_ln')(shortcut)

        x = nn.Conv(self.channels, kernel_size=(self.ksize, self.ksize), strides=self.strides, name='conv1')(x)
        x = nn.LayerNorm(name='ln1')(x)
        x = nn.relu(x)

        x = nn.Conv(self.channels, kernel_size=(self.ksize, self.ksize), strides=(1, 1), name='conv2')(x)
        x = nn.LayerNorm(name='ln2')(x)

        if self.use_se:
            x = SELayer(self.channels)(x)

        return nn.relu(x + shortcut)
