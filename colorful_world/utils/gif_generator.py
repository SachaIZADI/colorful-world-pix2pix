from PIL import Image
import os


def generate_gif(
        img_dir: str,
        gif_filename: str = "colorization_training.gif",
        milliseconds_per_frame: int = 20,
        sort_frames_per_epoch: bool = True,
):

    img_files = os.listdir(img_dir)
    img_files = [img_file for img_file in img_files if img_file.endswith(".png")]

    if sort_frames_per_epoch:
        img_files = sorted(img_files, key=lambda x: int(x.replace(".png", "").split("_")[-1]))

    frames = [Image.open(os.path.join(img_dir, img_file)) for img_file in img_files]

    frames[0].save(
        os.path.join(img_dir, gif_filename),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=len(frames) * milliseconds_per_frame,
        loop=0
    )


if __name__ == '__main__':
    generate_gif("/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/results/color_evolution")
