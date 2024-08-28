from torch import nn


class TransformationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_9 = 9
        kernel = 3
        self.num_residual_blocks = 5
        self.convolutional_downsampling = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_9,
                stride=1,
                padding=kernel_9 // 2,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel,
                stride=2,
                padding=kernel // 2,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                64,
                128,
                kernel,
                stride=2,
                padding=kernel // 2,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel, padding="same"),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel, padding="same"),
            nn.InstanceNorm2d(128, affine=True),
        )
        self.convolutional_upsampling = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel,
                stride=2,
                padding=kernel // 2,
                output_padding=1,
            ),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel, stride=2, padding=kernel // 2, output_padding=1
            ),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_9, stride=1, padding=kernel_9 // 2),
            nn.ReLU(),
            # nn.InstanceNorm2d(3, affine=True),
            # nn.Tanh(),
        )

    def forward(self, x):
        print(x.max())
        x = self.convolutional_downsampling(x)
        for _ in range(self.num_residual_blocks):
            residual = x
            x = self.residual_block(x)
            x += residual
        x = self.convolutional_upsampling(x)
        print(x.max())
        return x
