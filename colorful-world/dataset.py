import os
import torch
from torch.utils.data import Dataset
import cv2
from config import Config

config = Config()


class Dataset_Color(Dataset):
    '''
    Data pipeline of colored images to train the model
    '''

    def __init__(self, root_dir):
        """
        root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = os.path.join(self.root_dir, self.image_files[idx])
        clr, bw = self.generate_data(file, config.image_size)
        sample = {'clr': clr, 'bw': bw}
        return sample

    def generate_data(self, file, size):
        '''
        Convert RGB images to black and white (bw)
        '''

        img = cv2.imread(file)
        img = cv2.resize(img, (size, size))

        clr = img
        bw = cv2.cvtColor(clr, cv2.COLOR_BGR2GRAY)
        # Scale the images to [-1, 1] as suggested by the pix2pix authors
        clr = ((clr / 256) - 0.5) * 2.0
        bw = ((bw / 256) - 0.5) * 2.0
        clr = torch.from_numpy(clr).type(torch.FloatTensor).resize_(3, size, size)
        bw = torch.from_numpy(bw).type(torch.FloatTensor).resize_(1, size, size)

        return clr, bw


class Dataset_BW(Dataset):
    '''
    Data pipeline of black&white images to train the model
    '''

    def __init__(self, root_dir):
        """
        root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = os.path.join(self.root_dir, self.image_files[idx])
        clr, bw = self.generate_data(file, config.image_size)
        sample = {'clr': clr, 'bw': bw}
        return sample

    def generate_data(self, file, size):
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (size, size))
        bw = img
        # Scale the images to [-1, 1] as suggested by the pix2pix authors
        bw = ((bw / 256) - 0.5) * 2.0
        bw = torch.from_numpy(bw).type(torch.FloatTensor).resize_(1, size, size)
        return bw, bw