from PIL import Image
import os


def main():

    IMAGES_DIR = "/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/results/color_evolution"
    img_files = os.listdir(IMAGES_DIR)
    frames = [Image.open(os.path.join(IMAGES_DIR, img_file)) for img_file in img_files]
    frames[0].save(
        os.path.join(IMAGES_DIR, 'colorization_training.gif'),
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=len(frames)*20,
        loop=0
    )

if __name__ == '__main__':
    main()
