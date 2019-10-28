from colorful_world.models import cGAN
from colorful_world.config import Config


def main():
    config = Config()
    cgan = cGAN(config=config)
    train = True
    predict = True

    if train:
        print("---- Started to train")
        cgan.train()
    if predict:
        imgs = cgan.predict()
        for img in imgs:
            img.show()


if __name__ == "__main__":
    main()
