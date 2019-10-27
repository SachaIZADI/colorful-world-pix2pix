import torch
from config import Config
from model import GAN
from torch import optim
import torch.nn as nn

config  = Config()
model = GAN(config)
model.predict()