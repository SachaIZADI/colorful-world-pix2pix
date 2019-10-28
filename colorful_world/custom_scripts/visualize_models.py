from torchviz import make_dot
from colorful_world.models import Generator, Discriminator
from colorful_world.config import Config
from colorful_world.dataset import DatasetColorBW

def main():

    config = Config()
    generator = Generator()

    dataset_bw_clr = DatasetColorBW(
        root_dir=config.lfw_root_dir
    )
    sample = dataset_bw_clr.__getitem__(0)
    bw_tensor = sample["bw"]
    bw_tensor = bw_tensor.unsqueeze(0)
    colored_tensor = generator(bw_tensor)

    make_dot(colored_tensor).render(
        filename="generator",
        format="png",
        directory="/Users/sachaizadi/Documents/Projets/colorful-world/media",
        cleanup=True
    )

    discriminator = Discriminator(image_size=config.image_size)
    sample = dataset_bw_clr.__getitem__(0)

    prediction_discriminator = discriminator(
        clr=sample["clr"].unsqueeze(0),
        bw=sample["bw"].unsqueeze(0)
    )

    make_dot(prediction_discriminator).render(
        filename="discriminator",
        format="png",
        directory="/Users/sachaizadi/Documents/Projets/colorful-world/media",
        cleanup=True
    )


if __name__ == "__main__":
    main()