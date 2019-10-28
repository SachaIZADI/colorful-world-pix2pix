import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # --------- Encoder ---------
        # *bs = batch size
        # input = bs x 256 x 256 x 1  / output = bs x 128 x 128 x 64
        self.encod1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # input = bs x 128 x 128 x 64  / output = bs x 64 x 64 x 128
        self.encod2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128),
        )
        # input = bs x 64 x 64 x 128 / output = bs x 32 x 32 x 256
        self.encod3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 256 / output = bs x 16 x 16 x 512
        self.encod4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 512 / output = bs x 8 x 8 x 512
        self.encod5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 512 / output = bs x 4 x 4 x 512
        self.encod6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 512 / output = bs x 2 x 2 x 512
        self.encod7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 512 / output = bs x 1 x 1 x 512
        self.encod8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
        )

        # --------- Decoder ---------
        # input = bs x 1 x 1 x 512 / output = bs x 2 x 2 x 512
        self.decod8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 2*512 / output = bs x 4 x 4 x 512
        self.decod7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 2*512 / output = bs x 8 x 8 x 512
        self.decod6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 2*512 / output = bs x 16 x 16 x 512
        self.decod5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 2*512 / output = bs x 32 x 32 x 256
        self.decod4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 2*256 / output = bs x 64 x 64 x 128
        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
        )
        # input = bs x 32 x 32 x 2*128 / output = bs x 128 x 128 x 64
        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 128, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
        )
        # input = bs x 128 x 128 x 2*64 / output = bs x 256 x 256 x 3
        self.decodout = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 64, out_channels=3, kernel_size=4, padding=1, stride=2),
            nn.Tanh())

    def forward(self, x: torch.Tensor):
        # --------- Encoder ---------
        e1 = self.encod1(x)
        e2 = self.encod2(e1)
        e3 = self.encod3(e2)
        e4 = self.encod4(e3)
        e5 = self.encod5(e4)
        e6 = self.encod6(e5)
        e7 = self.encod7(e6)
        e8 = self.encod8(e7)

        # --------- Decoder ---------
        d8 = self.decod8(e8)
        d7 = self.decod7(torch.cat([d8, e7], 1))  # concatenating layers cf. U-net
        d6 = self.decod6(torch.cat([d7, e6], 1))  # concatenating layers cf. U-net
        d5 = self.decod5(torch.cat([d6, e5], 1))  # concatenating layers cf. U-net
        d4 = self.decod4(torch.cat([d5, e4], 1))  # concatenating layers cf. U-net
        d3 = self.decod3(torch.cat([d4, e3], 1))  # concatenating layers cf. U-net
        d2 = self.decod2(torch.cat([d3, e2], 1))  # concatenating layers cf. U-net

        out = self.decodout(torch.cat([d2, e1], 1))

        return out
