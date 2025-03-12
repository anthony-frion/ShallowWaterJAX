from flax import nnx
from flax import linen
import jax
import jax.numpy as jnp
import torch
import os
from pytorch_model import UNET2D

DEVICE = "cpu"


def load_model(model, filename="my_checkpoint.pth.tar", device="cpu"):
    # with torch.no_grad():
    assert os.path.isfile(filename), f"Error: {filename} not found"
    if os.path.isfile(filename):
        print("DEVICE on Loading model: ", device)
        checkpoint = torch.load(filename, map_location=torch.device(device))
        # checkpoint = torch.load(filename, map_location=torch.device('cuda'))
        # print("=> Loading model:", checkpoint)
        print("+++++++++++++++++++ LOADING trained model ++++++++++++++++++++++++++")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def get_checkpoint(filename="my_checkpoint.pth.tar", device="cpu"):
    # with torch.no_grad():
    assert os.path.isfile(filename), f"Error: {filename} not found"
    if os.path.isfile(filename):
        checkpoint = torch.load(
            filename, map_location=torch.device(device), weights_only=True
        )
        return checkpoint


def load_doubleconv_weights(jax_doubleconv, torch_state_dict, torch_prefix=""):
    patterns = [
        {
            "conv1_weight": f"{torch_prefix}conv.0.weight",
            "conv1_bias": f"{torch_prefix}conv.0.bias",
            "norm1_weight": f"{torch_prefix}conv.1.weight",
            "norm1_bias": f"{torch_prefix}conv.1.bias",
            "conv2_weight": f"{torch_prefix}conv.3.weight",
            "conv2_bias": f"{torch_prefix}conv.3.bias",
            "norm2_weight": f"{torch_prefix}conv.4.weight",
            "norm2_bias": f"{torch_prefix}conv.4.bias",
        },
    ]
    keys_found = {}
    for pattern in patterns:
        matches = sum(1 for key in pattern.values() if key in torch_state_dict)
        if matches > 0:
            keys_found.update(
                {k: v for k, v in pattern.items() if v in torch_state_dict}
            )
            if matches == 8:  # All keys found
                break

    if not keys_found:
        print(f"Warning: No matching keys found for {torch_prefix}")
        return jax_doubleconv

    if "conv1_weight" in keys_found:
        weight = torch_state_dict[keys_found["conv1_weight"]].detach().cpu().numpy()
        jax_doubleconv.conv1.kernel = jnp.array(
            weight.transpose(2, 3, 1, 0)
        )  # OIHW -> HWIO

    if "conv1_bias" in keys_found:
        bias = torch_state_dict[keys_found["conv1_bias"]].detach().cpu().numpy()
        jax_doubleconv.conv1.bias = jnp.array(bias)

    if "norm1_weight" in keys_found:
        weight = torch_state_dict[keys_found["norm1_weight"]].detach().cpu().numpy()
        jax_doubleconv.norm1.scale = jnp.array(weight)

    if "norm1_bias" in keys_found:
        bias = torch_state_dict[keys_found["norm1_bias"]].detach().cpu().numpy()
        jax_doubleconv.norm1.bias = jnp.array(bias)

    if "conv2_weight" in keys_found:
        weight = torch_state_dict[keys_found["conv2_weight"]].detach().cpu().numpy()
        jax_doubleconv.conv2.kernel = jnp.array(
            weight.transpose(2, 3, 1, 0)
        )  # OIHW -> HWIO

    if "conv2_bias" in keys_found:
        bias = torch_state_dict[keys_found["conv2_bias"]].detach().cpu().numpy()
        jax_doubleconv.conv2.bias = jnp.array(bias)

    if "norm2_weight" in keys_found:
        weight = torch_state_dict[keys_found["norm2_weight"]].detach().cpu().numpy()
        jax_doubleconv.norm2.scale = jnp.array(weight)

    if "norm2_bias" in keys_found:
        bias = torch_state_dict[keys_found["norm2_bias"]].detach().cpu().numpy()
        jax_doubleconv.norm2.bias = jnp.array(bias)

    return jax_doubleconv


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
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        features,
        kernel_size,
        stride,
        padding,
    ):
        self.num_layers = num_layers
        self.in_layers = in_channels
        self.out_channels = out_channels
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        rngs = nnx.Rngs(0)
        self.dynamics = nnx.Conv(
            in_features=in_channels[0],
            out_features=int(features[0] / 2),
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )
        self.statics = nnx.Conv(
            in_features=in_channels[1],
            out_features=int(features[0] / 2),
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

        self.encoder = []
        for i in range(num_layers):
            in_ch = features[i - 1] if i > 0 else features[0]
            out_ch = features[i]
            self.encoder.append(
                DoubleConv2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    num_groups=out_ch // 4,
                )
            )

        self.up = []
        for i in range(num_layers - 1, 0, -1):
            in_ch = features[i]
            out_ch = features[i - 1]
            self.up.append(
                nnx.ConvTranspose(
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=(2, 2),
                    strides=2,
                    rngs=rngs,
                    transpose_kernel=True,
                )
            )

        self.decoder = []
        for i in range(num_layers - 1, 0, -1):
            in_ch = features[i]
            out_ch = features[i - 1]
            self.decoder.append(
                DoubleConv2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    num_groups=out_ch // 4,
                )
            )

        self.output = nnx.Conv(
            in_features=features[0],
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

    def __call__(self, x, **kwargs):
        H = kwargs["H"]
        mask = kwargs["mask"]
        x_dynamics = self.dynamics(x)
        x_statics = self.statics(
            jnp.concatenate([H, mask], axis=-1)
        )  # I guess we need to change it to axis=-1 because flax has channel dim last.
        x = jnp.concatenate([x_dynamics, x_statics], axis=-1)

        skip_connections = []
        for i in range(self.num_layers):
            if i == 0:
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](
                    linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                )
            skip_connections.append(x)
        skip_connections = skip_connections[::-1][
            1:
        ]  # reverse the list and remove the first element
        for i in range(self.num_layers - 1):
            x = self.up[i](x)
            if (
                x.shape[1] != skip_connections[i].shape[1]
                or x.shape[2] != skip_connections[i].shape[2]
            ):
                diffY = skip_connections[i].shape[1] - x.shape[1]
                diffX = skip_connections[i].shape[2] - x.shape[2]
                pad_left = diffX // 2
                pad_right = diffX - pad_left
                pad_top = diffY // 2
                pad_bottom = diffY - pad_top
                x = jnp.pad(
                    x,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                )
            x = jnp.concatenate([x, skip_connections[i]], axis=-1)
            x = self.decoder[i](x)

        # Output part remains the same from original code
        outputs = self.output(x)
        return outputs

    def load_torch_weights(self):
        network_params = {
            "name": "UNET2D",
            "condi_net": False,
            "in_channels": [3, 2],
            "out_channels": [3],
            "features": [16, 32, 64, 128, 256],
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "norm_type": "GroupNorm",
            "activation": "SiLU",
            "num_layers": 5,
        }
        model = UNET2D(**network_params)
        filename = "model.pth.tar"
        model = load_model(model, filename).to(DEVICE)
        print(model)
        checkpoint = get_checkpoint(filename)
        model_state_dict = checkpoint["model_state_dict"]

        # Load encoder weights
        for i in range(self.num_layers):
            torch_prefix = f"encoder.{i}."
            self.encoder[i] = load_doubleconv_weights(
                self.encoder[i], model_state_dict, torch_prefix
            )

        # Load decoder weights
        for i in range(self.num_layers - 1):
            # In PyTorch, decoder layers are typically named in reverse order
            torch_prefix = f"decoder.{i}."
            self.decoder[i] = load_doubleconv_weights(
                self.decoder[i], model_state_dict, torch_prefix
            )

        # Load dynamics and statics conv weights
        if "dynamics.weight" in model_state_dict:
            weight = model_state_dict["dynamics.weight"].detach().cpu().numpy()
            self.dynamics.kernel = jnp.array(
                weight.transpose(2, 3, 1, 0)
            )  # OIHW -> HWIO
            if "dynamics.bias" in model_state_dict:
                self.dynamics.bias = jnp.array(
                    model_state_dict["dynamics.bias"].detach().cpu().numpy()
                )

        if "statics.weight" in model_state_dict:
            weight = model_state_dict["statics.weight"].detach().cpu().numpy()
            self.statics.kernel = jnp.array(
                weight.transpose(2, 3, 1, 0)
            )  # OIHW -> HWIO
            if "statics.bias" in model_state_dict:
                self.statics.bias = jnp.array(
                    model_state_dict["statics.bias"].detach().cpu().numpy()
                )

        # Load output conv weights
        key_prefix = f"output."
        if key_prefix + "weight" in model_state_dict:
            weight = model_state_dict[key_prefix + "weight"].detach().cpu().numpy()
            self.output.kernel = jnp.array(weight.transpose(2, 3, 1, 0))  # OIHW -> HWIO
        if key_prefix + "bias" in model_state_dict:
            self.output.bias = jnp.array(
                model_state_dict[key_prefix + "bias"].detach().cpu().numpy()
            )

        # Load up convs weights (for ConvTranspose layers)
        for i in range(self.num_layers - 1):
            key_prefix = f"up.{i}."
            if key_prefix + "weight" in model_state_dict:
                weight = model_state_dict[key_prefix + "weight"].detach().cpu().numpy()
                # ConvTranspose weight conversion (OIHW -> HWOI in JAX)
                self.up[i].kernel = jnp.array(weight.transpose(2, 3, 1, 0))
            if key_prefix + "bias" in model_state_dict:
                self.up[i].bias = jnp.array(
                    model_state_dict[key_prefix + "bias"].detach().cpu().numpy()
                )

        return self


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
        padding=padding,
    )

    # Create sample input data
    batch_size = 1
    height, width = 100, 100
    x = jnp.ones((1, height, width, in_channels[0]))  # Dynamic input
    H = jnp.ones((1, height, width, 1))  # Static terrain
    mask = jnp.ones((1, height, width, 1))  # Binary mask

    # Pass data through model
    output = model(x, H=H, mask=mask)
    model.load_torch_weights()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {out_channels}")

    return output


# Run the test
if __name__ == "__main__":
    test_unet()
