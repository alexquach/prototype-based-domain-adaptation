import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from models.proto_model import ProtoModel
from utils.plotting import plot_rows_of_images, plot_latent

class LatentTransition(nn.Module):
    def __init__(self, source_model, target_model, epochs=10):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.source_model = source_model
        self.target_model = target_model
        self.epoch = 0
        self.epochs = epochs

        # forward
        self.linear_layer_1 = nn.Linear(source_model.latent_dim, 256)
        self.linear_layer_2 = nn.Linear(256, target_model.latent_dim)
        self.transition_model = nn.Sequential(
            self.linear_layer_1,
            nn.ReLU(),
            self.linear_layer_2
        )

        # backward
        self.inverse_transition_model = nn.Sequential(
            nn.Linear(target_model.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, source_model.latent_dim)
        )

        self.transition_model = nn.Linear(source_model.latent_dim, target_model.latent_dim)
        self.optim = optim.Adam([
            *self.transition_model.parameters(),
            *self.inverse_transition_model.parameters()
        ])
        # self.optim = optim.Adam([
        #     *self.linear_layer_1.parameters(),
        #     *self.linear_layer_2.parameters(),
        # ])
        self.true_reconstruction = target_model.decoder(target_model.proto_layer.prototypes)

        self.to(self.dev)

    def fit(self):
        """
        Learns a linear mapping between source and target prototypes

        """
        while self.epoch < self.epochs:
            self.train()

            for i in range(500):
                transformed_source = self.transition_model(self.source_model.proto_layer.prototypes)
                # transformed_target = (self.target_model.proto_layer.prototypes - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T))
                transformed_target = self.inverse_transition_model(self.target_model.proto_layer.prototypes)

                source_proto_dist, source_feature_dist = self.source_model.proto_layer(transformed_source)
                source_min_proto_dist = ProtoModel.get_min(source_proto_dist)
                source_min_feature_dist = ProtoModel.get_min(source_feature_dist)

                target_proto_dist, target_feature_dist = self.target_model.proto_layer(transformed_target)
                target_min_proto_dist = ProtoModel.get_min(target_proto_dist)
                target_min_feature_dist = ProtoModel.get_min(target_feature_dist)
                

                loss_source = F.mse_loss(transformed_source, self.source_model.proto_layer.prototypes)
                loss_target = F.mse_loss(transformed_target, self.target_model.proto_layer.prototypes)
                loss_alignment = 0.01 * (source_min_proto_dist.mean() + source_min_feature_dist.mean() + target_min_proto_dist.mean() + target_min_feature_dist.mean())

                print(f'{self.epoch} + {loss_source} + {loss_target} + {loss_alignment}')
                loss_alignment.backward(retain_graph=True)
                loss_source.backward()
                loss_target.backward()

                self.optim.step()
                self.optim.zero_grad()

            print(source_min_proto_dist)
            print(source_min_feature_dist)
            print(target_min_proto_dist)
            print(target_min_feature_dist)
            self.epoch += 1

    def forward(self, latent_source):
        """
        Process:
            1. Transition latent_source to latent_target
            2. Calculate distances from each prototype in target space
            3. Generate predictions from prototype distances

        Returns:
            Distribution of Predictions across each of the prototypes
        """
        latent_source = latent_source.to(self.dev)
        latent_target = self.transition_model(latent_source)

        proto_distances_target, _ = self.target_model.proto_layer(latent_target)
        prediction = self.target_model.predictor(proto_distances_target)

        return prediction
    
    def forward_recons(self, latent_source):
        """
        Types:
            1. Converts source (latent space) to target (latent space)
            2. Snaps prototype in the source space, then converts to target latent space
            3. Snaps prototype in the target source, after it's converted to the target latent space.

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

    def loss_func(self, prediction, label):
        pred_loss = nn.CrossEntropyLoss()(prediction, label).mean()
        
        prediction_class = prediction.argmax(1)
        num_correct = (prediction_class == label).sum().item()
        class_accuracy = num_correct / len(prediction)

        return pred_loss, class_accuracy


    def evaluate(self, source_test_ds):
        """ Evaluates the classification loss for the source test dataset
        
        1. Uses source encoder
        2. Uses transition layer
        3. Evaluates prediction from target predictor

        """
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in source_test_ds:
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)
                latent_source = self.source_model.encoder(xb)
                prediction = self.__call__(latent_source)

                single_row = len(xb), *self.loss_func(prediction, yb)
                if batch_results is None:
                    batch_results = single_row
                else:
                    batch_results = np.vstack((batch_results, single_row))

        size_of_batches, class_loss, class_accuracy = batch_results.T
        
        batch_size = np.sum(size_of_batches)

        class_loss = np.sum(np.multiply(class_loss, size_of_batches)) / batch_size
        class_accuracy = np.sum(np.multiply(class_accuracy, size_of_batches)) / batch_size

        return class_loss, class_accuracy

    def test_decoder_visualization(self, path_name):
        """ Encoded by target_model, Decoded by target_model """
        decoded = self.target_model.decoder(self.target_model.proto_layer.prototypes)
        plot_rows_of_images([decoded], path_name)

    def visualize_transformed_source_prototype(self, path_name):
        """ Encoded by source_model, Converted by latent_transition, Decoded by target_model """
        transformed_latent = self.transition_model(self.source_model.proto_layer.prototypes)
        transformed_prototype, _, _ = self.forward_recons(self.source_model.proto_layer.prototypes)
        true_decoded = self.target_model.decoder(self.target_model.proto_layer.prototypes)

        print(F.mse_loss(transformed_latent, self.target_model.proto_layer.prototypes))
        print(F.mse_loss(transformed_prototype, true_decoded).mean())

        plot_rows_of_images([transformed_prototype], path_name)        
        plot_rows_of_images([true_decoded], "supposed_decoded.jpg")        

    def visualize_sample(self, test_dl, path_name, num_samples=10):
        input_, labels = next(iter(test_dl))
        input_ = input_[:num_samples].to(self.dev)
        labels = labels[:num_samples].to(self.dev)

        latent_source = self.source_model.encoder(input_)
        latent_target = self.transition_model(latent_source)
        reconstructions = self.target_model.decoder(latent_target)

        plot_rows_of_images([input_, reconstructions], path_name)

    def visualize_latent(self, latent, labels=None, savepath=None):
        plot_latent(latent, labels, savepath=savepath)