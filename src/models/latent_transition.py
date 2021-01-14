import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils.plotting import plot_rows_of_images

class LatentTransition(nn.Module):
    def __init__(self, source_model, target_model, epochs=10):
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

    def fit(self, source_train_dl):
        """
        Trains on source samples to optimize prototypes + transition + decoder?

        """
        while self.epoch < self.epochs:
            self.train()
            for xb, yb in source_train_dl:
                latent_source = self.source_model.encoder(xb)
                prediction = self.__call__(latent_source)
                
                self.loss_val, _ = self.loss_func(prediction, yb)
                print(f'{self.epoch} + {self.loss_val}')

                self.loss_val.backward()

                self.optim.step()
                self.optim.zero_grad()
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
        """ Evaluates the classification loss for the source test dataset, using the transition layer and target_decoder

        """
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in source_test_ds:
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