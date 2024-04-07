from torch import nn

class TransformationNetwork(nn.Module):
    def __init__(self):
        super(TransformationNetwork, self).__init__()
        kernel_9 = (9, 9)
        kernel = (3, 3)
        self.num_residual_blocks = 5
        self.convolutional_downsampling = nn.Sequential(
            nn.Conv2d(3, 32, kernel_9, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel, padding='same'),
            nn.BatchNorm2d(128),
        )
        self.convolutional_upsampling = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_9, stride=1, padding=4),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        inputs = self.convolutional_downsampling(x)
        for _ in range(self.num_residual_blocks):
            x = self.residual_block(inputs)
            x += inputs
            inputs = x
        x = self.convolutional_upsampling(inputs)
        return x
