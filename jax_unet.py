from flax import nnx
from flax import linen
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
        rngs = nnx.Rngs(0)
        self.conv1 = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=self.out_channels,
            out_features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=self.num_groups, num_features=self.out_channels, rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=self.num_groups, num_features=self.out_channels, rngs=rngs
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
        rngs = nnx.Rngs(0)
        self.dynamics = nnx.Conv(in_features=in_channels[0], out_features=int(features[0]/2), kernel_size=kernel_size, strides=stride, padding=padding, rngs=rngs)
        self.statics = nnx.Conv(in_features=in_channels[1], out_features=int(features[0]/2), kernel_size=kernel_size, strides=stride, padding=padding, rngs=rngs)

        self.encoder = []
        for i in range(num_layers):
            in_ch = features[i-1] if i > 0 else features[0]
            out_ch = features[i]
            self.encoder.append(DoubleConv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, strides=stride, padding=padding, num_groups=out_ch//4))
        
        self.up = []
        for i in range(num_layers-1, 0, -1):
            in_ch = features[i]
            out_ch = features[i-1]
            self.up.append(nnx.ConvTranspose(in_features=in_ch, out_features=out_ch, kernel_size=(2, 2), strides=2, rngs=rngs))
            
        self.decoder = []
        for i in range(num_layers-1, 0, -1):
            in_ch = features[i]
            out_ch = features[i-1]
            self.decoder.append(DoubleConv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, strides=stride, padding=padding, num_groups=out_ch//4)) 
        
        self.output = []
        for i in range(out_channels):
            self.output.append(nnx.Conv(in_features=features[0], out_features=1, kernel_size=1, strides=1, padding=0, rngs=rngs))  
        
    def __call__(self, x, **kwargs):
        H = kwargs['H']
        mask = kwargs['mask']
        x_dynamics = self.dynamics(x)
        x_statics = self.statics(jnp.concatenate([H, mask], axis=-1)) # I guess we need to change it to axis=-1 because flax has channel dim last.
        x = jnp.concatenate([x_dynamics, x_statics], axis=-1)

        skip_connections = []
        for i in range(self.num_layers):
            if i == 0:
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](linen.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
            skip_connections.append(x)
        skip_connections = skip_connections[::-1][1:] # reverse the list and remove the first element
        for i in range(self.num_layers-1):
            x = self.up[i](x)
            if x.shape[1] != skip_connections[i].shape[1] or x.shape[2] != skip_connections[i].shape[2]:
                diffY = skip_connections[i].shape[1]- x.shape[1]
                diffX = skip_connections[i].shape[2] - x.shape[2]
                pad_left = diffX // 2
                pad_right = diffX - pad_left
                pad_top = diffY // 2
                pad_bottom = diffY - pad_top
                x = jnp.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            x = jnp.concatenate([x, skip_connections[i]], axis=-1)
            x = self.decoder[i](x)
            
        # Output part remains the same from original code
        outputs = []
        for i in range(self.out_channels):
            outputs.append(self.output[i](x))
        return jnp.concatenate(outputs, axis=-1)


# Test code
def test_unet():
    # Initialize model parameters
    num_layers = 5
    in_channels = [3, 2]  # [dynamics_channels, statics_channels]
    out_channels = 3
    features = [16, 32, 64, 128, 256]
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    
    # Create model
    model = unet(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    # Create sample input data
    batch_size = 1
    height, width = 100, 100
    x = jnp.ones((1, height, width, in_channels[0]))  # Dynamic input
    H = jnp.ones((1, height, width, 1))  # Static terrain
    mask = jnp.ones((1, height, width, 1))  # Binary mask
    
    # Pass data through model
    output = model(x, H=H, mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {out_channels}")
    
    return output

# Run the test
if __name__ == "__main__":
    test_unet()