import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from models.proto_layer import ProtoLayer
from models.predictor import Predictor
from utils.plotting import plot_rows_of_images

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def preprocess_conv(x):
    return x.view(-1, 1, 28, 28)

class ProtoModel(nn.Module):
    def __init__(self, num_prototypes, hidden1_dim, hidden2_dim, latent_dim, num_classes, learning_rate, use_convolution = False):
        super(ProtoModel, self).__init__()

        # NN structure parameters
        self.input_dim = 784
        self.num_prototypes = num_prototypes
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Loss related parameters
        self.classification_weight = 10
        self.reconstruction_weight = 1
        self.close_to_proto_weight = 1
        self.proto_close_to_weight = 1

        # data for storing models
        self.epoch = 0

        if use_convolution:
            self.build_parts_conv()
        else:
            self.build_parts()

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)

    def build_parts(self):
        # Encoder
        self.encoder_layer1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.encoder_layer2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.encoder_layer3 = nn.Linear(self.hidden2_dim, self.latent_dim)
        self.encoder = nn.Sequential(
            Lambda(lambda x: x.view(-1, self.input_dim)),
            self.encoder_layer1,
            nn.ReLU(),
            self.encoder_layer2,
            self.encoder_layer3,
        )

        # Decoder
        self.decoder_layer2 = nn.Linear(self.latent_dim, self.hidden2_dim)
        self.decoder_layer1 = nn.Linear(self.hidden2_dim, self.hidden1_dim)
        self.recons_layer = nn.Linear(self.hidden1_dim, self.input_dim)
        self.decoder = nn.Sequential(
            self.decoder_layer2,
            self.decoder_layer1,
            nn.ReLU(),
            self.recons_layer,
            nn.Sigmoid(),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )

        # ProtoLayer
        self.proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        # Predictor
        self.predictor = Predictor(self.num_prototypes, None, self.num_classes)

    def build_parts_conv(self):
        # Encoder
        self.encoder_layer1 = nn.Conv2d(1, 32, 3)
        self.encoder_layer2 = nn.Conv2d(32, 64, 3)
        self.encoder_layer3 = nn.Linear(36864, 128)
        self.latent_layer = nn.Linear(128, self.latent_dim)
        self.encoder = nn.Sequential(
            Lambda(preprocess_conv),
            self.encoder_layer1,
            nn.ReLU(),
            self.encoder_layer2,
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1)),
            self.encoder_layer3,
            nn.ReLU(),
            self.latent_layer,
        )

        # Decoder
        self.decoder_layer2 = nn.Linear(self.latent_dim, 128)
        self.decoder_layer1 = nn.Linear(128, 128)
        self.recons_layer = nn.Linear(128, self.input_dim)
        self.decoder = nn.Sequential(
            self.decoder_layer2,
            self.decoder_layer1,
            nn.ReLU(),
            self.recons_layer,
            nn.Sigmoid(),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )

        # ProtoLayer
        self.proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        # Predictor
        self.predictor = Predictor(self.num_prototypes, None, self.num_classes)

    def build_parts_alt_conv(self):
        # TODO: look into implementing convolutional decoder and remove this alternative convolutional option
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = nn.Sequential(
            Lambda(preprocess_conv),
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )
        
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

        self.decoder = nn.Sequential(
            self.t_conv1,
            nn.ReLU(),
            self.t_conv2,
            nn.Sigmoid(),
        )
        
        # ProtoLayer
        self.proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        # Predictor
        self.predictor = Predictor(self.num_prototypes, None, self.num_classes)


    @staticmethod
    def get_min(x):
        return torch.min(x, dim=1).values

    def forward(self, input_):
        latent = self.encoder(input_)
        
        proto_distances, feature_distances = self.proto_layer(latent)
        min_proto_dist = ProtoModel.get_min(proto_distances)
        min_feature_dist = ProtoModel.get_min(feature_distances)
        
        recons = self.decoder(latent)

        prediction = self.predictor(proto_distances)

        return input_, recons, prediction, min_proto_dist, min_feature_dist

    def loss_func(self, input_, recons, prediction, min_proto_dist, min_feature_dist, label):

        recons_loss = F.mse_loss(input_, recons).mean()
        pred_loss = nn.CrossEntropyLoss()(prediction, label).mean()
        proto_dist_loss = torch.mean(min_proto_dist)
        feature_dist_loss = torch.mean(min_feature_dist)

        overall_loss = self.reconstruction_weight * recons_loss +\
                       self.classification_weight * pred_loss +\
                       self.proto_close_to_weight * proto_dist_loss +\
                       self.close_to_proto_weight * feature_dist_loss

        return overall_loss, len(input_), recons_loss, pred_loss, proto_dist_loss, feature_dist_loss
        
    def fit(self, epochs, train_dl, visualize_samples=False):
        while self.epoch < epochs:
            self.train()
            for xb, yb in train_dl:
                input_, recons, prediction, min_proto_dist, min_feature_dist = self.__call__(xb)
                
                self.loss_val, _, _, _, _, _ = self.loss_func(input_, recons, prediction, min_proto_dist, min_feature_dist, yb)
                print(f'{self.epoch} + {self.loss_val}')

                self.loss_val.backward()

                self.optim.step()
                self.optim.zero_grad()
            self.epoch += 1
            if visualize_samples:
                self.visualize_sample(train_dl, path_name=f'src/visualizations/conv_lr_adam_overall{self.epoch}.jpg')

    def evaluate(self, test_dl):
        """ Generates loss metric for the test dataset

        Args:
            test_dl (torch.utils.data.DataLoader): DataLoader for the test dataset
            overall_metric: Determines whether to return overall loss or individual loss metrics

        Returns:
            if overall_metric, returns the aggregated, weighted overall loss metric
            else, returns each of the components of the loss (recons_loss, pred_loss, proto_dist_loss, feature_dist_loss)
        """
        self.eval()
        # List comprehension to generate zipped pairs of batch-averaged losses and the size of batch
        with torch.no_grad():
            overall_loss, size_of_batches, recons_loss, pred_loss, proto_dist_loss, feature_dist_loss = zip(
                *[self.loss_func(input_, recons, prediction, min_proto_dist, min_feature_dist, yb) 
                for xb, yb in test_dl
                for input_, recons, prediction, min_proto_dist, min_feature_dist in [self.__call__(xb)]]
            )
        overall_loss = np.sum(np.multiply(overall_loss, size_of_batches)) / np.sum(size_of_batches)
        recons_loss = np.sum(np.multiply(recons_loss, size_of_batches)) / np.sum(size_of_batches)
        pred_loss = np.sum(np.multiply(pred_loss, size_of_batches)) / np.sum(size_of_batches)
        proto_dist_loss = np.sum(np.multiply(proto_dist_loss, size_of_batches)) / np.sum(size_of_batches)
        feature_dist_loss = np.sum(np.multiply(feature_dist_loss, size_of_batches)) / np.sum(size_of_batches)
        return overall_loss, recons_loss, pred_loss, proto_dist_loss, feature_dist_loss

    def save_model(self, path_name):
        """ Saves the model, optimizers, loss, and epoch

        More info:
            https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        print(f'Saving model to {path_name}')
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': self.loss_val,
            }, path_name)

    @staticmethod
    def load_model(path_name, num_protos, hidden1_dim, hidden2_dim, latent_dim, num_classes, learning_rate, use_convolution=False):
        """
        Note:
            If used for inference, make sure to set model.eval()
        """
        print(f'Loading model from {path_name}')
        checkpoint = torch.load(path_name)

        loaded_model = ProtoModel(num_protos, hidden1_dim, hidden2_dim, latent_dim, num_classes, learning_rate, use_convolution)

        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_model.epoch = checkpoint['epoch']
        loaded_model.loss_val = checkpoint['loss']

        return loaded_model

    def visualize_sample(self, test_dl, num_samples=10, path_name=None, show=True):
        input_, labels = next(iter(test_dl))
        input_ = input_[:num_samples]
        labels = labels[:num_samples]

        reconstructions = self.decoder(self.encoder(input_))

        plot_rows_of_images([input_, reconstructions], path_name, show=show)

    def visualize_latent(self, latent_vector, path_name=None, show=True):
        visualize_latent = self.decoder(latent_vector)

        plot_rows_of_images([visualize_latent], path_name, show=show)

    def visualize_prototypes(self, path_name=None, show=True):
        self.visualize_latent(self.proto_layer.prototypes, path_name, show)
    