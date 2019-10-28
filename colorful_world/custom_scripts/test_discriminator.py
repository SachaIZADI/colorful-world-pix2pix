from colorful_world.dataset import DatasetColorBW
from colorful_world.models import Discriminator
from colorful_world.config import Config


def main():

    config = Config()
    discriminator = Discriminator(image_size=config.image_size)

    dataset_bw_clr = DatasetColorBW(
        root_dir=config.lfw_root_dir
    )
    sample = dataset_bw_clr.__getitem__(0)

    prediction = discriminator(
        clr=sample["clr"].unsqueeze(0),
        bw=sample["bw"].unsqueeze(0)
    )

    print(f"discriminator_output.shape = {prediction.shape}")
    print(f"discriminator_output = {prediction}")


if __name__ == "__main__":
    main()
