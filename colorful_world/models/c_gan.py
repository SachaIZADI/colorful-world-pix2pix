from colorful_world.config import Config
from colorful_world.dataset import DatasetColorBW
from colorful_world.models import Discriminator, Generator


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt



class cGAN(object):

    def __init__(self, config: Config):
        self.config = config
        self.data_init()
        self.model_init()
        self.is_trained = False

    def data_init(self):
        self.training_dataset = DatasetColorBW(self.config.train_dir)
        self.training_data_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.prediction_dataset = DatasetColorBW(
            self.config.prediction_dir,
            colored=False,
            bw=True
        )

        self.prediction_data_loader = DataLoader(
            dataset=self.prediction_dataset,
            batch_size=1,
            shuffle=False
        )

    def model_init(self):
        self.dis_model = Discriminator(image_size=self.config.image_size)

        self.gen_model = Generator()
        self.gen_model = self.gen_model

        self.optimizer_dis = optim.Adam(self.dis_model.parameters(), lr=self.config.lr_dis)
        self.optimizer_gen = optim.Adam(self.gen_model.parameters(), lr=self.config.lr_gen)

        if self.config.use_L1_loss:
            self.L1_loss = nn.L1Loss()
            self.lambda_L1 = self.config.lambda_L1
        else:
            self.L1_loss = None
            self.lambda_L1 = 0

    # ------------------------------

    def train(self):
        return self.training(
            dis_model=self.dis_model, 
            gen_model=self.gen_model,
            data_loader=self.training_data_loader,
            dis_optimizer=self.optimizer_dis,
            gen_optimizer=self.optimizer_gen,
            n_epochs=self.config.n_epochs,
            L1_loss=self.L1_loss,
            lambda_L1=self.lambda_L1
        )

    def training(
            self,
            dis_model: Discriminator, gen_model: Generator,
            data_loader: DataLoader,
            dis_optimizer: torch.optim, gen_optimizer: torch.optim,
            n_epochs: int = 1,
            L1_loss: nn.L1Loss = None, lambda_L1: float = 1.
    ):

        EPS = 1e-12

        use_gpu = self.config.gpu
        if use_gpu:
            torch.cuda.set_device(0)
            dis_model = dis_model.cuda()
            gen_model = gen_model.cuda()
            if L1_loss:
                L1_loss = L1_loss.cuda()

        dis_model.train(True)
        gen_model.train(True)

        dis_loss = np.zeros(n_epochs)
        gen_loss = np.zeros(n_epochs)

        if self.config.show_color_evolution:
            dataset_color_bw = DatasetColorBW(self.config.train_dir)
            _, bw_example = dataset_color_bw.generate_data(
                file=self.config.picture_color_evolution,
                img_size=self.config.image_size,
                colored=True,
                bw=True,
            )
            bw_example = bw_example.unsqueeze(0)
            if use_gpu:
                bw_example = bw_example.cuda()

        t = 0

        for epoch_num in range(n_epochs):

            dis_running_loss = 0.0
            gen_running_loss = 0.0
            size = 0

            for data in data_loader:

                clr_img, bw_img = data['clr'], data['bw']

                if use_gpu:
                    clr_img = clr_img.cuda()
                    bw_img = bw_img.cuda()

                batch_size = clr_img.size(0)
                size += batch_size

                dis_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                if t % 2 == 1:
                    dis_model.train(False)
                    gen_model.train(True)
                    Gx = gen_model(bw_img)  # Generates fake colored images

                else:
                    dis_model.train(True)
                    gen_model.train(False)
                    Gx = gen_model(bw_img).detach()  # Detach the generated images for training the discriminator only

                Dx = dis_model(clr_img, bw_img)  # Produces probabilities for real images
                Dg = dis_model(Gx, bw_img)  # Produces probabilities for generator images

                d_loss = -torch.mean(
                    torch.log(Dx + EPS) + torch.log(1. - Dg + EPS))  # Loss function of the discriminator.
                g_loss = - torch.mean(torch.log(Dg + EPS))  # Loss function of the generator.

                if L1_loss:
                    g_loss = g_loss + lambda_L1 * L1_loss(Gx, clr_img)

                # Run backprop and update the weights of the Generator accordingly
                dis_running_loss += d_loss.data.cpu().numpy()
                if t % 2 == 0:
                    d_loss.backward()
                    dis_optimizer.step()

                # Run backprop and update the weights of the Discriminator accordingly
                gen_running_loss += g_loss.data.cpu().numpy()
                if t % 2 == 1:
                    g_loss.backward()
                    gen_optimizer.step()

                t += 1

            epoch_dis_loss = dis_running_loss / size
            epoch_gen_loss = gen_running_loss / size
            dis_loss[epoch_num] = epoch_dis_loss
            gen_loss[epoch_num] = epoch_gen_loss

            print('Train - Discriminator Loss: {:.4f} Generator Loss: {:.4f}'.format(epoch_dis_loss, epoch_gen_loss))

            if self.config.show_color_evolution and t % 2 == 1:
                Gx_example = gen_model(bw_example).detach()
                Gx_example_img = Image.fromarray(
                    np.uint8((Gx_example[0].permute(1, 2, 0).numpy() / 2 + 0.5) * 256)
                )
                Gx_example_img.save(
                    fp=os.path.join(self.config.result_dir, "color_evolution", f"Gx_epoch_{epoch_num}.png"),
                    format="png"
                )

            if (epoch_num % 10 == 0 and epoch_num != 0) or epoch_num == n_epochs-1:
                torch.save(gen_model, os.path.join(self.config.model_dir, f'gen_model_{epoch_num}.pk'))
                torch.save(dis_model, os.path.join(self.config.model_dir, f'dis_model_{epoch_num}.pk'))
                print("Saved Model")

        if self.config.plot_loss:
            fig = plt.figure()
            plt.plot(list(range(n_epochs)), dis_loss, label="discriminator")
            plt.plot(list(range(n_epochs)), gen_loss, label="generator")
            plt.title("Evolution of the Discriminator and Generator loss during the training")
            plt.grid()
            plt.legend(loc='upper right')
            plt.show()
            fig.savefig(os.path.join(self.config.result_dir, "loss_graph.png"), format="png")


        self.is_trained = True

        return cGAN

    # ------------------------------

    def predict(self):
        if not self.is_trained:
            # Load a model with which to make the prediction
            self.predict_generator = torch.load(
                os.path.join(self.config.model_dir, 'gen_model_%s.pk' % str(self.config.n_epochs - 1))
            )
        else:
            self.predict_generator = self.gen_model

        self.predict_generator.eval()
        return self.predicting(self.predict_generator, self.prediction_data_loader)


    def predicting(self, gen_model, data_loader):

        use_gpu = self.config.gpu
        if use_gpu:
            torch.cuda.set_device(0)

        gen_model.eval()

        imgs = []

        for data in data_loader:
            bw_img = data['bw']

            if use_gpu:
                bw_img = bw_img.cuda()

            fake_img = gen_model(bw_img).detach()

            for i in range(len(fake_img)):
                img_array = fake_img.cpu().numpy()[i].transpose(1, 2, 0)
                img_array = (((img_array / 2.0 + 0.5) * 256).astype('uint8'))
                img = Image.fromarray(img_array)
                imgs.append(img)

        return imgs
