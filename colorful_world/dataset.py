import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from colorful_world.config import Config

config = Config()


class DatasetColorBW(Dataset):

    def __init__(self, root_dir: str, colored: bool = True, bw: bool = True):
        self.root_dir = root_dir
        self.image_files = os.listdir(self.root_dir)
        self.colored = colored
        self.bw = bw

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file = os.path.join(self.root_dir, self.image_files[idx])

        clr, bw = self.generate_data(
            file=file,
            img_size=config.image_size,
            colored=self.colored,
            bw=self.bw
        )

        sample = {'clr': clr, 'bw': bw}

        if not self.colored: sample.pop("clr")
        if not self.bw: sample.pop("bw")

        return sample

    def generate_data(self, file: str, img_size: int, colored: bool, bw: bool):

        img_clr = Image.open(file)
        img_clr = img_clr.resize((img_size, img_size))

        if colored:
            img_clr_array = np.array(img_clr)
            # Scale the images to [-1, 1]
            img_clr_array = ((img_clr_array / 256) - 0.5) * 2.0
            # tensor shape 3 (channels) x img_size x img_size
            img_clr_tensor = torch.from_numpy(img_clr_array).type(torch.FloatTensor).permute(2, 0, 1)

        else:
            img_clr_tensor = None

        if bw:
            img_bw = img_clr.convert('L')
            img_bw_array = np.array(img_bw)
            # Scale the images to [-1, 1]
            img_bw_array = ((img_bw_array / 256) - 0.5) * 2.0
            # tensor shape 1 (channel) x img_size x img_size
            img_bw_tensor = torch.from_numpy(img_bw_array).type(torch.FloatTensor).unsqueeze(0)

        else:
            img_bw_tensor = None

        return img_clr_tensor, img_bw_tensor
