import torch

class Config(object):

    def __init__(
            self,
            lr_dis=0.001,
            lr_gen=0.01,
            n_epochs=10,
            batch_size=4,
            use_L1_loss=True,
            lambda_L1=1,
            image_size=512,
            train_dir='/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/data/lfw_small',
            model_dir='/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/models/models_saved',
            test_dir='/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/data/lfw_small',
            prediction_dir='/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/data/lfw_small',
            predicted_dir='...',
            save_frequency=10,
            gpu=torch.cuda.is_available(),
            lfw_root_dir="/Users/sachaizadi/Documents/Projets/colorful-world/colorful_world/data/lfw",
    ):

        self.lr_dis = lr_dis
        self.lr_gen = lr_gen
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.use_L1_loss = use_L1_loss
        self.lambda_L1 = lambda_L1

        self.image_size = image_size

        self.model_dir = model_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.prediction_dir = prediction_dir
        self.predicted_dir = predicted_dir
        self.save_frequency = save_frequency

        self.gpu = gpu

        self.lfw_root_dir = lfw_root_dir