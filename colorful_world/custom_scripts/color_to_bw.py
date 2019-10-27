from colorful_world.dataset import DatasetColorBW
from colorful_world.config import Config
import numpy as np
from PIL import Image


def main():

    config = Config()

    dataset_bw_clr = DatasetColorBW(
        root_dir=config.lfw_root_dir
    )

    print("------ Dataset with colored and black & white pictures ------")
    print(f"len(dataset) = {len(dataset_bw_clr)}")
    sample = dataset_bw_clr.__getitem__(0)
    print(f"bw_tensor.shape = {sample['bw'].shape}")
    print(f"clr_tensor.shape = {sample['clr'].shape}")

    img_bw = Image.fromarray(
        (sample['bw'].numpy()/2 + 0.5) * 256
    )
    img_bw.show()

    img_clr = Image.fromarray(
        np.uint8((sample['clr'].numpy() / 2 + 0.5) * 256)
    )
    img_clr.show()


if __name__ == "__main__":
    main()