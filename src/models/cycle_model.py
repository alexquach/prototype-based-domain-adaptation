import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from models.proto_model import ProtoModel
from utils.plotting import plot_rows_of_images

class CycleModel(nn.Module):
    def __init__(self, source_model, target_model, epochs=10, weights=(1,1,1,1,1,1)):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.source_model = source_model
        self.target_model = target_model
        self.epoch = 0
        self.epochs = epochs

        # Transfer layer
        # Currently using linear
        self.transition_model = nn.Linear(source_model.latent_dim, target_model.latent_dim)

        self.weight_recon_source, self.weight_recon_target, self.weight_autoencode_source,\
            self.weight_autoencode_target, self.weight_class_source, self.weight_class_target = weights

        self.optim = optim.Adam([
            *self.source_model.parameters(),
            *self.target_model.parameters(),
            *self.transition_model.parameters()
        ])

        self.to(self.dev)

    def forward_source(self, xb_source):
        latent_source = self.source_model.encoder(xb_source)
        latent_target = self.transition_model(latent_source)

        # Get prediction from source
        proto_distances_source, _ = self.source_model.proto_layer(latent_source)
        prediction_source = self.source_model.predictor(proto_distances_source)
        
        transfer_recon_target = self.target_model.decoder(latent_target)
        transfer_latent_target = self.target_model.encoder(transfer_recon_target)

        # inverse linear transfer (target -> source)
        transfer_latent_source = (transfer_latent_target - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T))
        transfer_recon_source = self.source_model.decoder(transfer_latent_source)

        return xb_source, transfer_recon_source, prediction_source

    def forward_target(self, xb_target):
        latent_target = self.target_model.encoder(xb_target)
        latent_source = (latent_target - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T))

        proto_distances_target, _ = self.target_model.proto_layer(latent_target)
        prediction_target = self.target_model.predictor(proto_distances_target)
        
        transfer_recon_source = self.source_model.decoder(latent_source)
        transfer_latent_source = self.source_model.encoder(transfer_recon_source)

        # inverse linear transfer (target -> source)
        transfer_latent_target = self.transition_model(transfer_latent_source)
        transfer_recon_target = self.target_model.decoder(transfer_latent_target)

        return xb_target, transfer_recon_target, prediction_target

    def autoencode(self, xb_source, xb_target):
        latent_source = self.source_model.encoder(xb_source)
        latent_target = self.target_model.encoder(xb_target)

        recon_source = self.source_model.decoder(latent_source)
        recon_target = self.target_model.decoder(latent_target)

        return xb_source, recon_source, xb_target, recon_target

    def fit_combined_loss(self, source_train_dl, target_train_dl):
        """
        Trains using a combined loss for simultaneous optimization

        """

        while self.epoch < self.epochs:
            self.train()    

            # Loop over target_train and source_train datasets
            for (xb_target, yb_target), (xb_source, yb_source) in zip(target_train_dl, source_train_dl):
                xb_source = xb_source.to(self.dev)
                xb_target = xb_target.to(self.dev)
                yb_source = yb_source.to(self.dev)
                yb_target = yb_target.to(self.dev)

                # Forward pass for all components
                _, recon_source, prediction_source = self.forward_source(xb_source)
                _, recon_target, prediction_target = self.forward_target(xb_target)

                # 1. Loss on transfered source
                loss_recon_source = self.loss_recon(xb_source, recon_source)
                print(f'transfer source {self.epoch}: {loss_recon_source}')

                # 2. Loss on transfered target
                loss_recon_target = self.loss_recon(xb_target, recon_target)
                print(f'transfer target {self.epoch}: {loss_recon_target}')

                # 3 + 4. Loss on autoencoded 
                _, autoencode_source, _, autoencode_target = self.autoencode(xb_source, xb_target) 
                loss_autoencode_source = self.loss_recon(xb_source, autoencode_source)
                loss_autoencode_target = self.loss_recon(xb_target, autoencode_target)
                print(f'autoencode {self.epoch}: {loss_autoencode_source} + {loss_autoencode_target}')

                # 5 + 6. Loss on straight-through classification error
                loss_class_source, acc_source = self.loss_pred(prediction_source, yb_source)
                loss_class_target, acc_target = self.loss_pred(prediction_target, yb_target)
                print(f'class loss {self.epoch}: {loss_class_source} + {loss_class_target}')
                print(f'class acc {self.epoch}: {acc_source} + {acc_target}')

                # calculate combined loss
                self.loss_combined = self.weight_recon_source * loss_recon_source +\
                                     self.weight_recon_target * loss_recon_target +\
                                     self.weight_autoencode_source * loss_autoencode_source +\
                                     self.weight_autoencode_target * loss_autoencode_target +\
                                     self.weight_class_source * loss_class_source +\
                                     self.weight_class_target * loss_class_target
                self.loss_combined.backward()

                self.optim.step()
                self.optim.zero_grad()

            self.epoch += 1

    def loss_pred(self, prediction, label):
        pred_loss = nn.CrossEntropyLoss()(prediction, label).mean()
        
        prediction_class = prediction.argmax(1)
        num_correct = (prediction_class == label).sum().item()
        class_accuracy = num_correct / len(prediction)

        return pred_loss, class_accuracy

    def loss_recon(self, input_, recon_image):
        return F.mse_loss(input_.view(input_.shape[0], -1), recon_image).mean()

    def evaluate(self, target_test_dl, eval_model=None):
        """ Evaluates the classification loss for the target test dataset

        """
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in target_test_dl:
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)

                if eval_model:
                    input_, recon_image, prediction, _, _ = eval_model(xb)
                else:
                    input_, recon_image, prediction, _, _ = self.target_model(xb)

                single_row = len(xb), *self.loss_pred(prediction, yb)
                if batch_results is None:
                    batch_results = single_row
                else:
                    batch_results = np.vstack((batch_results, single_row))

        size_of_batches, class_loss, class_accuracy = batch_results.T
        
        batch_size = np.sum(size_of_batches)

        class_loss = np.sum(np.multiply(class_loss, size_of_batches)) / batch_size
        class_accuracy = np.sum(np.multiply(class_accuracy, size_of_batches)) / batch_size

        return class_loss, class_accuracy

    def save_model(self, path_name):
        """ Saves the model, optimizers, loss, and epoch

        More info:
            https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        print(f'Saving model to {path_name}')
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'loss_weights': (self.weight_recon_source, self.weight_recon_target, self.weight_autoencode_source,\
                            self.weight_autoencode_target, self.weight_class_source, self.weight_class_target)
            }, path_name)

    @staticmethod
    def load_model(path_name, model_1, model_2):
        """
        Note:
            If used for inference, make sure to set model.eval()
        """
        print(f'Loading model from {path_name}')
        if torch.cuda.is_available():
            checkpoint = torch.load(path_name)
        else:
            checkpoint = torch.load(path_name, map_location=torch.device('cpu'))

        # configure loss weights if exist
        if checkpoint['loss_weights']:
            loaded_model = CycleModel(model_1, model_2, weights=checkpoint['loss_weights'])
        else:
            loaded_model = CycleModel(model_1, model_2)

        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.epoch = checkpoint['epoch']

        return loaded_model

    def visualize_prototypes(self, path_name=None):
        """ Visualizes the prototypes of both source and target models """

        source_proto = self.source_model.decoder(self.source_model.proto_layer.prototypes)
        target_proto = self.target_model.decoder(self.target_model.proto_layer.prototypes)

        plot_rows_of_images([source_proto, target_proto], path_name)
    
    def visualize_samples(self, source_dl, target_dl, path_name=None):
        """ Visualizes the samples of both source and target models """
        xb_source, _ = next(iter(source_dl))
        xb_target, _ = next(iter(target_dl))
        xb_source = xb_source.to(self.dev)[:10]
        xb_target = xb_target.to(self.dev)[:10]

        # latent side
        latent_source = self.source_model.encoder(xb_source)
        latent_target = self.target_model.encoder(xb_target)

        # reconstructions
        recon_source = self.source_model.decoder(latent_source)
        recon_target = self.target_model.decoder(latent_target)

        _, recon_transfer_source, _ = self.forward_source(xb_source)
        _, recon_transfer_target, _ = self.forward_target(xb_target)

        plot_rows_of_images([xb_source, xb_target, recon_source, recon_target, recon_transfer_source, recon_transfer_target], path_name)