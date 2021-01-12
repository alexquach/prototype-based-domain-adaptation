import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import funcy
from collections.abc import Iterable

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
    def __init__(self, config, learning_rate):
        super(ProtoModel, self).__init__()
        self.config = config

        # NN structure parameters
        self.input_dim = funcy.get_in(self.config, ['input_dim'], None)
        self.hidden_layers = funcy.get_in(self.config, ["hidden_layers"], None)
        self.hidden_activations = funcy.get_in(self.config, ["hidden_activations"], None)
        self.recon_activation = funcy.get_in(self.config, ["recon_activation"], None)
        self.latent_dim = funcy.get_in(self.config, ['latent_dim'], None)
        self.num_prototypes = funcy.get_in(self.config, ['num_prototypes'], None)
        self.num_classes = funcy.get_in(self.config, ['num_classes'], None)
        self.use_convolution = funcy.get_in(self.config, ['use_convolution'], None)

        # Loss related parameters
        self.classification_weight = 10
        self.reconstruction_weight = 1
        self.close_to_proto_weight = 1
        self.proto_close_to_weight = 1

        # data for storing models
        self.epoch = 0

        if self.use_convolution:
            self.build_parts_conv()
        else:
            self.build_parts()

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)

    def process_activation(self, activation, layers):
        if activation:
            if isinstance(activation, Iterable):
                for activation_part in activation:
                    if activation_part:
                        layers.append(activation_part)
            else:
                layers.append(activation)
        return layers

    def build_network(self, layer_dims, activations):
        layers = []

        for i in range(len(layer_dims) - 1):
            layers = self.process_activation(activations[i], layers)

            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

        if activations[len(layer_dims) - 1]:
            layers = self.process_activation(activations[len(layer_dims) - 1], layers)

        return nn.Sequential(*layers)


    def build_parts(self):
        # Encoder
        encoder_dims = [
            self.config['input_dim'],
            *self.config['hidden_layers'],
            self.config['latent_dim']
        ]
        encoder_activations = [
            Lambda(lambda x: x.view(-1, self.input_dim)), 
            *self.config['hidden_activations'], 
            funcy.get_in(self.config, ['latent_activation'], None),
        ]

        # Decoder
        decoder_dims = list(reversed(encoder_dims))
        decoder_activations = [
            None,
            *list(reversed(self.config['hidden_activations'])), 
            (funcy.get_in(self.config, ['recon_activation'], None), Lambda(lambda x: x.view(-1, self.input_dim))), 
        ]

        self.encoder = self.build_network(encoder_dims, encoder_activations)
        self.decoder = self.build_network(decoder_dims, decoder_activations)

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
        
    def fit(self, epochs, train_dl, visualize_sample_name=None):
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
            if visualize_sample_name:
                self.visualize_sample(train_dl, path_name=f'src/visualizations/{visualize_sample_name}_{self.epoch}.jpg')

    def class_accuracy(self, prediction, label):
        """ Calculates class accuracy given prediction and labels

        Args:
            prediction (tensor): tensor with some distribution per sample (batchsize, num_classes)
            label (tensor): tensor with ground truth class index labels 

        Returns:
            float with accuracy score
        """
        prediction_class = prediction.argmax(1)

        num_correct = (prediction_class == label).sum().item()
        class_accuracy = num_correct / len(prediction)
        return class_accuracy

    def evaluate(self, test_dl):
        """ Generates loss metrics for the test dataset

        Args:
            test_dl (torch.utils.data.DataLoader): DataLoader for the test dataset

        Returns:
            Returns the overall loss, class accuracy, reconstruction loss, prediction loss, prototype distance loss, and feature distance loss
        """
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in test_dl:
                for input_, recons, prediction, min_proto_dist, min_feature_dist in [self.__call__(xb)]:
                    single_row = *self.loss_func(input_, recons, prediction, min_proto_dist, min_feature_dist, yb), self.class_accuracy(prediction, yb)
                    if batch_results is None:
                        batch_results = single_row
                    else:
                        batch_results = np.vstack((batch_results, single_row))

        overall_loss, size_of_batches, recons_loss, pred_loss, proto_dist_loss, feature_dist_loss, class_acc = batch_results.T
        
        batch_size = np.sum(size_of_batches)

        overall_loss = np.sum(np.multiply(overall_loss, size_of_batches)) / batch_size
        recons_loss = np.sum(np.multiply(recons_loss, size_of_batches)) / batch_size
        pred_loss = np.sum(np.multiply(pred_loss, size_of_batches)) / batch_size
        proto_dist_loss = np.sum(np.multiply(proto_dist_loss, size_of_batches)) / batch_size
        feature_dist_loss = np.sum(np.multiply(feature_dist_loss, size_of_batches)) / batch_size
        class_acc = np.sum(np.multiply(class_acc, size_of_batches)) / batch_size
        return overall_loss, class_acc, recons_loss, pred_loss, proto_dist_loss, feature_dist_loss

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
    def load_model(path_name, config, learning_rate):
        """
        Note:
            If used for inference, make sure to set model.eval()
        """
        print(f'Loading model from {path_name}')
        checkpoint = torch.load(path_name)

        loaded_model = ProtoModel(config, learning_rate)

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
    
    def generate_latent_transition(self, target_model):
        latent_transition = LatentTransition(self, target_model)
        latent_transition.test_decoder_visualization("before_decoder_train.jpg")
        latent_transition.fit()
        latent_transition.test_decoder_visualization("after_decoder_train.jpg")
        return latent_transition

class LatentTransition(nn.Module):
    def __init__(self, source_model, target_model, epochs=1000):
        super().__init__()
        self.source_model = source_model
        self.target_model = target_model
        self.epoch = 0
        self.epochs = epochs
        self.linear_layer_1 = nn.Linear(source_model.latent_dim, 256)
        self.linear_layer_2 = nn.Linear(256, target_model.latent_dim)
        self.transition_model = nn.Sequential(
            self.linear_layer_1,
            nn.ReLU(),
            self.linear_layer_2
        )
        self.optim = optim.Adam([
            *self.linear_layer_1.parameters(),
            *self.linear_layer_2.parameters(),
        ])
        self.true_reconstruction = target_model.decoder(target_model.proto_layer.prototypes)

    def fit(self):
        while self.epoch < self.epochs:
            self.train()
            # training on 10 prototypes
            recons, recons_before, recons_after = self.__call__(self.source_model.proto_layer.prototypes)
            
            self.loss_val = F.mse_loss(recons, self.true_reconstruction).mean()
            print(f'{self.epoch} + {self.loss_val}')

            self.loss_val.backward(retain_graph=True)

            self.optim.step()
            self.optim.zero_grad()
            self.epoch += 1

    def forward(self, latent_source):
        """
        Types:
            1. Converts source (latent space) to target (latent space)
            2. Snaps prototype in the source space, then converts to target latent space
            3. Snaps prototype in the target source, after it's converted to the target latent space.

        Returns:
            1. latent_source -> latent_target -> decoded
            2. latent_source -> prototype_s -> prototype_s_target -> decoded (proto before transition)
            3. latent_source -> latent_target -> prototype_t -> decoded (proto after transition)
        """

        # 1
        latent_target = self.transition_model(latent_source)

        # 2
        proto_distances_source, _ = self.source_model.proto_layer(latent_source)
        prototype_s_index = torch.argmax(self.source_model.predictor(proto_distances_source), dim=1)
        prototype_s = self.source_model.proto_layer.prototypes[prototype_s_index]
        prototype_s_target = self.transition_model(prototype_s)

        # 3
        proto_distances_target, _ = self.target_model.proto_layer(latent_target)
        prototype_t_index = torch.argmax(self.target_model.predictor(proto_distances_target), dim=1)
        prototype_t = self.target_model.proto_layer.prototypes[prototype_t_index]

        return self.target_model.decoder(latent_target), self.target_model.decoder(prototype_s_target), self.target_model.decoder(prototype_t)

    def evaluate(self, test_dl):
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in test_dl:
                encoded_input = self.source_model.encoder(xb)
                for no_proto, before_proto, after_proto in [self.__call__(encoded_input)]:
                    single_row = len(xb), F.mse_loss(xb, no_proto).mean(), F.mse_loss(xb, before_proto).mean(), F.mse_loss(xb, after_proto).mean()
                    if batch_results is None:
                        batch_results = single_row
                    else:
                        batch_results = np.vstack((batch_results, single_row))

        size_of_batches, no_proto, before_proto, after_proto = batch_results.T
        
        batch_size = np.sum(size_of_batches)

        no_proto = np.sum(np.multiply(no_proto, size_of_batches)) / batch_size
        before_proto = np.sum(np.multiply(before_proto, size_of_batches)) / batch_size
        after_proto = np.sum(np.multiply(after_proto, size_of_batches)) / batch_size

        return no_proto, before_proto, after_proto

    def test_decoder_visualization(self, path_name):
        """ Encoded by target_model, Decoded by target_model """
        decoded = self.target_model.decoder(self.target_model.proto_layer.prototypes)
        plot_rows_of_images([decoded], path_name)

    def visualize_transformed_source_prototype(self, path_name):
        """ Encoded by source_model, Converted by latent_transition, Decoded by target_model """
        transformed_prototype, _, _ = self.__call__(self.source_model.proto_layer.prototypes)
        true_decoded = self.target_model.decoder(self.target_model.proto_layer.prototypes)

        print(F.mse_loss(transformed_prototype, true_decoded).mean())

        plot_rows_of_images([transformed_prototype], path_name)

    def visualize_sample(self, test_dl, path_name, num_samples=10):
        input_, labels = next(iter(test_dl))
        input_ = input_[:num_samples]
        labels = labels[:num_samples]

        encoded = self.source_model.encoder(input_)
        reconstructions, _, _ = self.__call__(encoded)

        plot_rows_of_images([input_, reconstructions], path_name)