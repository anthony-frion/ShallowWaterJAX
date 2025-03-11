from flax import nnx
import jax
import jax.numpy as jnp


class DoubleConv2D(nnx.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, strides, padding, num_groups
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_groups = num_groups
        self.conv1 = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.conv2 = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.norm1 = nnx.GroupNorm(
            num_features=self.in_channels, num_groups=self.num_groups
        )
        self.norm2 = nnx.GroupNorm(
            num_features=self.in_channels, num_groups=self.num_groups
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.silu(x)
        return x


class unet(nnx.Module):
    def __init__(self, num_layers, in_channels, out_channels, features, kernel_size, stride, padding):
        self.num_layers = num_layers
        self.in_layers = in_channels
        self.out_channels = out_channels
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dynamics = nnx.Conv(in_features=in_channels[0], out_features=int(features[0]/2), kernel_size=kernel_size, strides=stride, padding=padding)
        self.statics = nnx.Conv(in_features=in_channels[1], out_features=int(features[0]/2), kernel_size=kernel_size, strides=stride, padding=padding)

        self.encoder = []
        for i in range(num_layers):
            in_ch = features[i-1] if i > 0 else features[0]
            out_ch = features[i]
            self.encoder.append(DoubleConv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, strides=stride, padding=padding, num_groups=out_ch//4))
        
        self.pool = nnx.MaxPool(kernel_size=2, strides=2)
        self.up = []
        for i in range(num_layers-1, 0, -1):
            in_ch = features[i]
            out_ch = features[i-1]
            self.up.append(nnx.ConvTranspose(in_features=in_ch, out_features=out_ch, kernel_size=2, strides=2))
            self.up.append(DoubleConv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, strides=stride, padding=padding, num_groups=out_ch//4))
        
        self.output = []
        for i in range(out_channels):
            self.output.append(nnx.Conv(in_features=features[0], out_features=1, kernel_size=1, strides=1, padding=0))  
        
    def __call__(self, x):
        pass