from config import Config
from dataset import Dataset_Color, Dataset_BW
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import cv2


class Generator(nn.Module):
    '''
    As suggested by the authors of pix2pix (https://arxiv.org/pdf/1611.07004.pdf), the generator is a U-net (https://arxiv.org/pdf/1505.04597.pdf), i.e. a variant of an auto-encoder
    where layers are symmetrically stacked one to the other.
    It learns a mapping between gray scale to RGB space, conditionally to a grayscale image.
    '''

    def __init__(self):
        super(Generator, self).__init__()
        # --------- Encoder ---------
        # *bs = batch size
        # input = bs x 256 x 256 x 1  / output = bs x 128 x 128 x 64
        self.encod1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        # input = bs x 128 x 128 x 64  / output = bs x 64 x 64 x 128
        self.encod2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )
        # input = bs x 64 x 64 x 128 / output = bs x 32 x 32 x 256
        self.encod3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 256 / output = bs x 16 x 16 x 512
        self.encod4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 512 / output = bs x 8 x 8 x 512
        self.encod5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 512 / output = bs x 4 x 4 x 512
        self.encod6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 512 / output = bs x 2 x 2 x 512
        self.encod7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 512 / output = bs x 1 x 1 x 512
        self.encod8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )

        # --------- Decoder ---------
        # input = bs x 1 x 1 x 512 / output = bs x 2 x 2 x 512
        self.decod8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 2*512 / output = bs x 4 x 4 x 512
        self.decod7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 2*512 / output = bs x 8 x 8 x 512
        self.decod6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 2*512 / output = bs x 16 x 16 x 512
        self.decod5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 2*512 / output = bs x 32 x 32 x 256
        self.decod4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 2*256 / output = bs x 64 x 64 x 128
        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
        )
        # input = bs x 32 x 32 x 2*128 / output = bs x 128 x 128 x 64
        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 128, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
        )
        # input = bs x 128 x 128 x 2*64 / output = bs x 256 x 256 x 3
        self.decodout = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 64, out_channels=3, kernel_size=4, padding=1, stride=2),
            nn.Tanh())

    def forward(self, x):
        # --------- Encoder ---------
        e1 = self.encod1(x)
        e2 = self.encod2(e1)
        e3 = self.encod3(e2)
        e4 = self.encod4(e3)
        e5 = self.encod5(e4)
        e6 = self.encod6(e5)
        e7 = self.encod7(e6)
        e8 = self.encod8(e7)

        # --------- Decoder ---------
        d8 = self.decod8(e8)
        d7 = self.decod7(torch.cat([d8, e7], 1))  # concatenating layers cf. U-net
        d6 = self.decod6(torch.cat([d7, e6], 1))  # concatenating layers cf. U-net
        d5 = self.decod5(torch.cat([d6, e5], 1))  # concatenating layers cf. U-net
        d4 = self.decod4(torch.cat([d5, e4], 1))  # concatenating layers cf. U-net
        d3 = self.decod3(torch.cat([d4, e3], 1))  # concatenating layers cf. U-net
        d2 = self.decod2(torch.cat([d3, e2], 1))  # concatenating layers cf. U-net

        out = self.decodout(torch.cat([d2, e1], 1))

        return out


# TODO: here

class Discriminator(nn.Module):
    '''
    The discriminator is a classical ConvNet classifier.
    It learns to discriminate fake-colored image vs. originally colored images conditionally to the black & white image,
    conditionally to a b&w image
    '''

    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size

        self.conv = nn.Sequential(
            # input = bs x 256 x 256 x (1+3) / output = bs x 128 x 128 x 64
            nn.Conv2d(in_channels=1 + 3, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            # input = bs x 128 x 128 x 64 / output = bs x 64 x 64 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            # input = bs x 64 x 64 x 128 / output = bs x 32 x 32 x 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            # input = bs x 32 x 32 x 256 / output = bs x 16 x 16 x 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),

            # input = bs x 16 x 16 x 512 / output = bs x 8 x 8 x 1
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
        )

        # TODO: I'm not sure about the last layers in the discriminator

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.image_size / (2 ** 5)) * int((self.image_size / (2 ** 5))) * 1,
                      out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, clr, bw):
        conc_clr_bw = torch.cat((clr, bw), 1)
        result = self.conv(conc_clr_bw)
        result = result.view(-1, int(self.image_size / (2 ** 5)) * int(self.image_size / (2 ** 5) * 1))
        result = self.fc(result)
        result = torch.sigmoid(result)
        return result


class GAN(object):
    '''
    The Generative Adversarial Network framework we used to train our color generator
    '''

    def __init__(self, config):
        self.config = config
        self.data_init()
        self.model_init()

    def data_init(self):
        '''
        Initialize the data pipeline to train the model
        '''
        self.training_dataset = Dataset_Color(self.config.train_dir)
        self.training_data_loader = DataLoader(dataset=self.training_dataset, batch_size=self.config.batch_size,
                                               shuffle=True)

        # self.testing_dataset = Dataset_Color(self.config.test_dir)
        # self.testing_data_loader = DataLoader(dataset= self.testing_dataset, batch_size= self.config.batch_size, shuffle=False)

        self.prediction_dataset = Dataset_BW(self.config.prediction_dir)
        self.prediction_data_loader = DataLoader(dataset=self.prediction_dataset, batch_size=1, shuffle=False)

    def model_init(self):
        self.dis_model = Discriminator(image_size=self.config.image_size)
        self.dis_model = self.dis_model
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

    def train(self):
        return self.training(dis_model=self.dis_model, gen_model=self.gen_model,
                             data_loader=self.training_data_loader,
                             dis_optimizer=self.optimizer_dis, gen_optimizer=self.optimizer_gen,
                             n_epochs=self.config.n_epochs,
                             L1_loss=self.L1_loss, lambda_L1=self.lambda_L1)

    def training(self, dis_model, gen_model, data_loader,
                 dis_optimizer, gen_optimizer, n_epochs=1,
                 L1_loss=None, lambda_L1=1):
        '''
        Training loop for the GAN
        '''

        EPS = 1e-12

        use_gpu = torch.cuda.is_available()
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

        t = 0

        for epoch_num in range(n_epochs):

            dis_running_loss = 0.0
            gen_running_loss = 0.0
            size = 0

            for data in data_loader:
                # colored images, black & white images
                # NB: colored image are real images (not generated ones)
                clr_img, bw_img = data['clr'], data['bw']

                if use_gpu:
                    clr_img = clr_img.cuda()
                    bw_img = bw_img.cuda()

                batch_size = clr_img.size(0)
                size += batch_size

                dis_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                # TODO : detach ?
                if t % 2 == 0:
                    Gx = gen_model(bw_img).detach()  # Generates fake colored images
                else:
                    Gx = gen_model(bw_img)
                Dx = dis_model(clr_img, bw_img)  # Produces probabilities for real images
                Dg = dis_model(Gx, bw_img)  # Produces probabilities for generator images

                d_loss = -torch.mean(
                    torch.log(Dx + EPS) + torch.log(1. - Dg + EPS))  # Loss function of the discriminator.
                g_loss = -torch.mean(torch.log(Dg + EPS))  # Loss function of the generator.

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

            if (epoch_num + 1) % 1 == 0 and epoch_num != 0:
                print(
                    'Train - Discriminator Loss: {:.4f} Generator Loss: {:.4f}'.format(epoch_dis_loss, epoch_gen_loss))

            # Save the model on the disk for the future
            if (epoch_num + 1) % self.config.save_frequency == 0 and epoch_num != 0:
                if not os.path.exists(self.config.model_dir):
                    os.makedirs(self.config.model_dir)
                torch.save(gen_model, self.config.model_dir + 'gen_model_%i.pk' % epoch_num)
                torch.save(dis_model, self.config.model_dir + 'dis_model_%i.pk' % epoch_num)
                print("Saved Model")

            """
            def test(self):
                self.test_model = torch.load(self.config.model_dir + 'model_weights_epoch_9.pk')
                self.test_generator = self.test_model['gen_model_state_dict']
                self.test_discriminator = self.test_model['dis_model_state_dict']
                return self.testing(self.test_generator, self.test_discriminator, self.testing_data_loader,  self.loss, L1_loss=None, lambda_L1=1)

            def testing(self, dis_model, gen_model, data_loader, loss_fn, L1_loss=None, lambda_L1=1):
                dis_model.train(False)
                gen_model.train(False)

                dis_running_loss = 0.0
                gen_running_loss = 0.0
                size = 0

                for data in data_loader:

                    clr_img, bw_img = data['clr'], data['bw']

                    if use_gpu:
                        clr_img = clr_img.cuda()
                        bw_img  = bw_img.cuda()

                    batch_size = clr_img.size(0)
                    size += batch_size

                    fake_img = gen_model(bw_img).detach()
                    fake_out = dis_model(fake_img)

                    loss = loss_fn(fake_out, torch.ones(batch_size, 1).cuda())

                    if L1_loss is not None :
                        loss = loss + lambda_L1 * L1_loss()

                    gen_running_loss += loss_fn(fake_out, torch.ones(batch_size, 1).cuda())
                    dis_running_loss += loss_fn(fake_out, torch.zeros(batch_size, 1).cuda())

                    print('Test - Discriminator Loss: {:.4f} Generator Loss: {:.4f}'.format(dis_running_loss / size,
                                                                                            gen_running_loss / size))

                    for data in data_loader:

                        clr_img, bw_img = data['clr'], data['bw']

                        if use_gpu == True:
                            clr_img = clr_img.cuda()
                            bw_img  = bw_img.cuda()

                        batch_size = clr_img.size(0)
                        size += batch_size

                        out = dis_model(clr_img)

                        loss = loss_fn(out, torch.ones(batch_size, 1).cuda()) 
                        dis_running_loss += loss.data.cpu().numpy()

                        fake_img = gen_model(bw_img).detach()
                        fake_out = dis_model(fake_img)
                        loss = loss_fn(fake_out, torch.zeros(batch_size, 1).cuda())
                        dis_running_loss += loss.data.cpu().numpy()


                        fake_img = gen_model(bw_img).detach()
                        fake_out = dis_model(fake_img)
                        loss = loss_fn(fake_out, torch.ones(batch_size, 1).cuda())
                        gen_running_loss += loss.data.cpu().numpy()

                        if L1_loss is not None :
                            loss = loss + lambda_L1 * L1_loss()
            """

    def predict(self):
        '''
        Generate colored images from an initial b&w image
        '''
        # Load a model with which to make the prediction
        self.predict_generator = torch.load(self.config.model_dir + 'gen_model_%s.pk' % str(self.config.n_epochs - 1))
        self.predict_generator.eval()

        return self.predicting(self.predict_generator, self.prediction_data_loader)

    def predicting(self, gen_model, data_loader):

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.set_device(0)

        # gen_model.train(False)

        for data in data_loader:
            bw_img = data['bw']
            batch_size = bw_img.size(0)

            if use_gpu:
                bw_img = bw_img.cuda()

            fake_img = gen_model(bw_img).detach()

            # print(fake_img.cpu().numpy().shape)

            for i in range(len(fake_img)):
                img = fake_img.cpu().numpy()[i].transpose(1, 2, 0)
                img = (((img / 2.0 + 0.5) * 256).astype('uint8'))
                '''
                red = img[:, :, 2].copy()
                blue = img[:, :, 1].copy()
                green = img[:, :, 0].copy()
                img[:, :, 0] = red
                img[:, :, 1] = blue
                img[:, :, 2] = green
                '''
                cv2.imwrite(self.config.predicted_dir + '%i.jpg' % i, img)