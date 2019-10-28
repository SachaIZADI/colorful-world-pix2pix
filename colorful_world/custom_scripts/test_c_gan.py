from colorful_world.models import cGAN
from colorful_world.config import Config


def main():
    config = Config()
    cgan = cGAN(config=config)


if __name__ == "__main__":
    main()
