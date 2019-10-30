import os
import random
from shutil import copyfile


def generate_sample_lfw(size: int, src_folder: str, dst_folder: str):
    imgs = os.listdir(src_folder)
    nb_imgs = len(imgs)
    selected_imgs = [imgs[i] for i in random.sample(range(1, nb_imgs), size)]

    for img in selected_imgs:
        copyfile(
            src=os.path.join(src_folder, img),
            dst=os.path.join(dst_folder, img)
        )


def main():
    src_folder = 'lfw'

    # TODO: remove the "#" (idem in data.sh)
    dst_folder_small = 'lfw_small'
    lfw_small_size = 20
    #generate_sample_lfw(size=lfw_small_size, src_folder=src_folder, dst_folder=dst_folder_small)

    dst_folder_medium = 'lfw_medium'
    lfw_medium_size = 1000
    generate_sample_lfw(size=lfw_medium_size, src_folder=src_folder, dst_folder=dst_folder_medium)


if __name__ == "__main__":
    main()
