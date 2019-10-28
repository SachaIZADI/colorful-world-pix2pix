import os
import random
from shutil import copyfile


def main():
    folder = 'lfw'
    new_folder = 'lfw_small'
    lfw_small_size = 20

    imgs = os.listdir(folder)
    nb_imgs = len(imgs)
    selected_imgs = [imgs[i] for i in random.sample(range(1, nb_imgs), lfw_small_size)]

    for img in selected_imgs:
        copyfile(
            src=os.path.join(folder, img),
            dst=os.path.join(new_folder, img)
        )


if __name__ == "__main__":
    main()
