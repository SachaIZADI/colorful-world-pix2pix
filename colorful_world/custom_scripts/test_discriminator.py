from colorful_world.dataset import DatasetColorBW
from colorful_world.models import Discriminator
from colorful_world.config import Config
import numpy as np
from PIL import Image


def main():

    config = Config()
    discriminator = Discriminator(image_size=config.image_size[0])

    dataset_bw_clr = DatasetColorBW(
        root_dir=config.lfw_root_dir
    )
    sample = dataset_bw_clr.__getitem__(0)

    print(sample["bw"].shape)
    print(sample["clr"].shape)

    prediction = discriminator(
        clr=sample["clr"],
        bw=sample["bw"]
    )

    print(prediction.shape)



if __name__ == "__main__":
    main()
