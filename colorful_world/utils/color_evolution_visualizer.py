from colorful_world.models import cGAN
from colorful_world.config import Config
import os


def visualize_color_evolution_training(
        prediction_dir="/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/data/to_predict",
        result_dir="/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/results/",
        model_dir="/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/models/models_saved"
):

    config = Config(
        prediction_dir=prediction_dir,
        result_dir=result_dir,
        model_dir=model_dir
    )

    models_saved = [model for model in os.listdir(model_dir) if model.endswith(".pk") and model.startswith("gen_model")]

    c_gan = cGAN(config=config)

    for i in range(len(c_gan.prediction_dataset)):
        try:
            os.mkdir(path=os.path.join(config.result_dir, "color_evolution", str(i)))
        except:
            raise Warning("Could not mkdir...")

    for model in models_saved:

        colored_imgs = c_gan.predict(path_to_model=os.path.join(model_dir, model))
        epoch_num = model.replace(".pk", "").split("_")[-1]

        i = 0
        for img in colored_imgs:
            img.save(fp=os.path.join(config.result_dir, "color_evolution", str(i), f"Gx_epoch_{epoch_num}.png"),)
            i += 1


if __name__ == '__main__':
    visualize_color_evolution_training()