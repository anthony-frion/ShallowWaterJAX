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


class Upsample(nnx.Module):
    out_dim: int
    is_deconv: bool

    # ConvTranspose Parameters
    kernel_size: int = 2
    strides: int = 2

    @nn.compact
    def forward(self, inputs1, inputs2):

        if self.is_deconv:
            outputs2 = nn.ConvTranspose(
                features=self.out_dim,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(inputs2)
        else:
            outputs2 = nn.UpsamplingBilinear2d(scale_factor=2)(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]

        padding = 2 * [offset // 2, offset // 2]

        outputs1 = jnp.pad(inputs1, padding)

        return unetConv(features=self.out_dim, is_batchnorm=False)(
            jnp.concatenate([outputs1, outputs2], 1)
        )


class unet(nn.Module):
    feature_scale: int = 4
    n_classes: int = 21
    is_deconv: bool = True
    use_batchnorm: bool = True
    kernel_size: int = 2

    @nn.compact
    def __call__(self, x):

        is_deconv = self.is_deconv
        use_batchnorm = self.use_batchnorm
        feature_scale = self.feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # downsampling

        conv1 = unetConv(filters[0], use_batchnorm)(x)
        maxpool1 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv1)

        conv2 = unetConv(filters[1], use_batchnorm)(maxpool1)
        maxpool2 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv2)

        conv3 = unetConv(filters[2], use_batchnorm)(maxpool2)
        maxpool3 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv3)

        conv4 = unetConv(filters[3], use_batchnorm)(maxpool3)
        maxpool4 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv4)

        center = unetConv(filters[4], use_batchnorm)(maxpool4)

        # upsampling
        up4 = Upsample(filters[3], is_deconv=is_deconv)(conv4, center)
        up3 = Upsample(filters[2], is_deconv=is_deconv)(conv3, up4)
        up2 = Upsample(filters[1], is_deconv=is_deconv)(conv2, up3)
        up1 = Upsample(filters[0], is_deconv=is_deconv)(conv1, up2)

        # final conv (without any concat)
        final = nn.Conv(self.n_classes, 1)(up1)

        return final


if __name__ == "__main__":
    pass
