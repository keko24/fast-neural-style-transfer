from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        kernel_9 = (9, 9)
        kernel = (3, 3)
        self.num_residual_blocks = 3
        self.convolutional_downsampling = nn.Sequential(
            nn.Conv2d(3, 32, kernel_9, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel),
            nn.BatchNorm2d(128),
        )
        self.convolutional_upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_9),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        input = self.convolutional_downsampling(x)
        for _ in range(self.num_residual_blocks):
            x = self.residual_block(input)
            x += input
            input = x
        x = (self.convolutional_upsampling(x) + 1) * 255
        return x
