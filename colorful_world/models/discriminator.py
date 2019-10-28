import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size

        self.conv = nn.Sequential(
            # input = bs x 256 x 256 x (1+3) / output = bs x 128 x 128 x 64
            nn.Conv2d(in_channels=1+3, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),

            # input = bs x 128 x 128 x 64 / output = bs x 64 x 64 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128),

            # input = bs x 64 x 64 x 128 / output = bs x 32 x 32 x 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256),

            # input = bs x 32 x 32 x 256 / output = bs x 16 x 16 x 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),

            # input = bs x 16 x 16 x 512 / output = bs x 8 x 8 x 1
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.image_size / (2 ** 5)) * int((self.image_size / (2 ** 5))) * 1,
                      out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=1),
        )


    # TODO: check the forward method

    def forward(self, clr, bw):
        cat_clr_bw = torch.cat((clr, bw), 1)
        print(cat_clr_bw.shape)
        features = self.conv(cat_clr_bw)
        flatten = features.view(-1, int(self.image_size / (2 ** 5)) * int(self.image_size / (2 ** 5) * 1))
        result = self.fc(flatten)
        output = torch.sigmoid(result)
        return output
