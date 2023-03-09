import flax.linen as nn
import jax.numpy as jnp

class ResLayer(nn.Module):
    kernel_size: int
    num_channels: int
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self) -> None:

        self.conv1 = nn.Conv(
            self.num_channels, 
            kernel_size = (self.kernel_size, self.kernel_size),
            padding = "same",
            strides = self.strides
        )
        self.bn1 = nn.BatchNorm(not self.training)

        self.conv2 = nn.Conv(
            self.num_channels, 
            kernel_size = (self.kernel_size, self.kernel_size),
            padding = "same",
            strides = self.strides 
        )

        self.bn2 = nn.BatchNorm(not self.training)

        self.conv3 = nn.Conv(
            self.num_channels,
            kernel_size = (1,1),
            strides = self.strides
        )

    def __call__(self, X:jnp.array) -> jnp.array:
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(X))
        Y += X
        return nn.relu(X)

class SELayer(nn.Module):
    num_channels: int
    reduction: int = 16

    def setup(self) -> None:
        self.fc = nn.Sequential(
            nn.Dense(self.num_channels, self.num_channels // 16),
            nn.relu(),
            nn.Dense(self.num_channels // 16, self.num_channels),
            nn.sigmoid()
        )
    
    def __call__(self, X: jnp.array) -> jnp.array:
        b, _, _, c= X.shape
        Y = jnp.mean(X, axis = [1, 2], keepdims=False)
        Y = self.fc(Y)
        Y = jnp.reshape(Y, (b, 1, 1, c))
        Y = jnp.broadcast_to(Y, X.shape)
        return Y * X 

class ResSEBlock(nn.Module):
    kernel_size: int
    num_channels: int 
    strides: tuple
    training: bool 
    reduction: int
    out_channels:int

    def setup(self) -> None:

        self.resLayer = ResLayer(
            kernel_size = self.kernel_size, 
            num_channels = self.num_channels,
            strides = self.strides,
            training = self.training
        )

        self.seLayer = SELayer(
            num_channels = self.num_channels,
            reduction = self.reduction
        )

        self.finConv = nn.Conv(self.out_channels, strides=(1, 1))

    def __call__(self, X:jnp.array) -> jnp.array:
        X = self.resLayer(X)
        X = self.seLayer(X)
        X = self.finConv(X)
        return X
