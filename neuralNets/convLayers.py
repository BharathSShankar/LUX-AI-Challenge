import haiku as hk
import jax
import jax.numpy as jnp

class SELayer(hk.Module):
    """Squeeze-excitation layer for ResNet."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def __call__(self, x):
        x_global_avg = jnp.mean(x, axis=(1, 2), keepdims=True)
        x = hk.Linear(self.channels // 16, name='fc1')(x_global_avg)
        x = jax.nn.relu(x)
        x = hk.Linear(self.channels, name='fc2')(x)
        x = jax.nn.sigmoid(x)
        return x * x_global_avg

class ResNetBlock(hk.Module):
    """Residual block for ResNet."""

    def __init__(self, channels, strides=(1, 1), use_projection=True, use_se=False, ksize=3):
        super().__init__()
        self.channels = channels
        self.strides = strides
        self.use_projection = use_projection
        self.use_se = use_se
        self.ksize = ksize

    def __call__(self, x):
        shortcut = x

        # Projection shortcut in case input and output shapes are different
        if self.use_projection:
            shortcut = hk.Conv2D(self.channels, kernel_shape=(1, 1), stride=self.strides, name='shortcut_conv')(shortcut)
            shortcut = hk.BatchNorm(name='shortcut_bn')(shortcut)

        x = hk.Conv2D(self.channels, kernel_shape=(self.ksize, self.ksize), stride=self.strides, name='conv1')(x)
        x = hk.BatchNorm(name='bn1')(x)
        x = jax.nn.relu(x)

        x = hk.Conv2D(self.channels, kernel_shape=(self.ksize, self.ksize), stride=(1, 1), name='conv2')(x)
        x = hk.BatchNorm(name='bn2')(x)

        if self.use_se:
            x = SELayer(self.channels)(x)

        return jax.nn.relu(x + shortcut)