from colorful_world.dataset import DatasetColorBW
from colorful_world.models import Generator
from colorful_world.config import Config
import numpy as np
from PIL import Image


def main():

    config = Config()
    generator = Generator()

    dataset_bw_clr = DatasetColorBW(
        root_dir=config.lfw_root_dir
    )
    sample = dataset_bw_clr.__getitem__(0)
    bw_tensor = sample["bw"]
    bw_tensor = bw_tensor.unsqueeze(0).unsqueeze(0)

    print(f"input_tensor_shape = {bw_tensor.shape}")
    output = generator(bw_tensor)
    print(f"output_tensor_shape = {output.shape}")

    img_bw = Image.fromarray(
        (sample['bw'].numpy()/2 + 0.5) * 256
    )
    img_bw.show()

    img_clr = Image.fromarray(
        np.uint8((output[0].permute(1, 2, 0).detach().numpy() / 2 + 0.5) * 256)
    )
    img_clr.show()


if __name__ == "__main__":
    main()
