from colorful_world.config import Config
from colorful_world.models import cGAN


config = Config()
model = cGAN(config)
model.train()
