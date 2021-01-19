import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils.plotting import plot_rows_of_images

class TransferModel(nn.Module):
    def __init__(self, source_model, target_model, epochs=10):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.source_model = source_model
        self.target_model = target_model
        self.epoch = 0
        self.epochs = epochs

        # Transfer layer
        self.transfer_layer_1 = nn.Linear(source_model.latent_dim, 256)
        self.transfer_layer_2 = nn.Linear(256, target_model.latent_dim)
        self.transfer_layer = nn.Sequential(
            self.transfer_layer_1,
            nn.ReLU(),
            self.transfer_layer_2,
        )

        # Optimizers
        self.optim_target_unlabel = optim.Adam([
            *self.target_model.parameters(),
        ])
        self.optim_transfer_samples = optim.Adam([
            self.target_model.proto_layer.prototypes,
        ])
        self.optim_target_label = optim.Adam([
            *self.target_model.parameters(),
            self.target_model.proto_layer.prototypes,
        ])
        self.optim_transfer_layer = optim.Adam([
            *self.transfer_layer_1.parameters(),
            *self.transfer_layer_2.parameters(),
        ])

        self.true_reconstruction = target_model.decoder(target_model.proto_layer.prototypes)

        self.to(self.dev)


    def fit_interleave(self, source_train_dl, target_train_dl):
        """
        Trains on source samples to optimize prototypes + transition + decoder?

        """
        label_loss_history = []
        label_acc_history = []

        # TODO: Check for even distribution of labelled examples
        xb_label, yb_label = next(iter(target_train_dl))
        xb_label = xb_label.to(self.dev)
        yb_label = yb_label.to(self.dev)

        while self.epoch < self.epochs:
            self.train()    

            # Loop over target_train and source_train datasets
            for (xb_target_unlabel, _), (xb_source, yb_source) in zip(target_train_dl, source_train_dl):
                xb_target_unlabel = xb_target_unlabel.to(self.dev)
                xb_source = xb_source.to(self.dev)
                yb_source = yb_source.to(self.dev)

                # Train on unlabelled target
                input_, recon_image, _, _, _ = self.target_model(xb_target_unlabel)

                self.loss_target_unlabel = self.loss_recon(input_, recon_image)
                print(f'unlabelled target {self.epoch}: {self.loss_target_unlabel}')
                self.loss_target_unlabel.backward()
                self.optim_target_unlabel.step()
                self.optim_target_unlabel.zero_grad()


                # Train on transfered source
                latent_source = self.source_model.encoder(xb_source)
                latent_target = self.transfer_layer(latent_source)
                proto_distances_target, _ = self.target_model.proto_layer(latent_target)
                prediction = self.target_model.predictor(proto_distances_target)

                self.loss_transfer_samples, _ = self.loss_pred(prediction, yb_source)
                print(f'transfer source {self.epoch}: {self.loss_transfer_samples}')
                self.loss_transfer_samples.backward()
                self.optim_transfer_samples.step()
                self.optim_transfer_samples.zero_grad()


                # Train on few-shot labelled target batch
                input_, recon_image, prediction, _, _ = self.target_model(xb_label)

                self.loss_target_label, target_label_accuracy = self.loss_pred(prediction, yb_label)
                print(f'labelled target {self.epoch}: {self.loss_target_label} and acc {target_label_accuracy}')
                self.loss_target_label.backward()
                self.optim_target_label.step()
                self.optim_target_label.zero_grad()

                label_loss_history.append(self.loss_target_label)
                label_acc_history.append(target_label_accuracy)


                # Train transfer layer
                recon_target_proto = self.transfer_layer(self.source_model.proto_layer.prototypes)
                self.loss_transfer_layer = self.loss_recon(recon_target_proto, self.target_model.proto_layer.prototypes)
                print(f'transfer layer {self.epoch}: {self.loss_transfer_layer}')
                self.loss_transfer_layer.backward()

                self.optim_transfer_layer.step()
                self.optim_transfer_layer.zero_grad()

            self.epoch += 1

        print(label_loss_history)
        print(label_acc_history)
        print(yb_label.bincount())

    def fit_epoch_based(self, source_train_dl, target_train_dl):
        """
        Trains on source samples to optimize prototypes + transition + decoder?

        """
        label_loss_history = []
        label_acc_history = []
        while self.epoch < self.epochs:
            self.train()    

            # Train on unlabelled target
            for xb, _ in target_train_dl:
                xb = xb.to(self.dev)

                input_, recon_image, _, _, _ = self.target_model(xb)

                self.loss_target_unlabel = self.loss_recon(input_, recon_image)
                print(f'unlabelled target {self.epoch}: {self.loss_target_unlabel}')
                self.loss_target_unlabel.backward()

                self.optim_target_unlabel.step()
                self.optim_target_unlabel.zero_grad()

            # Train on transfered source
            for xb, yb in source_train_dl:
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)

                latent_source = self.source_model.encoder(xb)
                latent_target = self.transfer_layer(latent_source)
                proto_distances_target, _ = self.target_model.proto_layer(latent_target)
                prediction = self.target_model.predictor(proto_distances_target)
                
                self.loss_transfer_samples, _ = self.loss_pred(prediction, yb)
                print(f'transfer source {self.epoch}: {self.loss_transfer_samples}')
                self.loss_transfer_samples.backward()

                self.optim_transfer_samples.step()
                self.optim_transfer_samples.zero_grad()

            # TODO: Make consistent batch, not random batch
            # Train on few-shot labelled target
            # Currently using one batch of labelled data
            for i in range(100):
                xb, yb = next(iter(target_train_dl))
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)
                input_, recon_image, prediction, _, _ = self.target_model(xb)

                self.loss_target_label, target_label_accuracy = self.loss_pred(prediction, yb)
                print(f'labelled target {self.epoch}: {self.loss_target_label} and acc {target_label_accuracy}')
                self.loss_target_label.backward()

                self.optim_target_label.step()
                self.optim_target_label.zero_grad()
            label_loss_history.append(self.loss_target_label)
            label_acc_history.append(target_label_accuracy)
            print(label_loss_history)
            print(label_acc_history)


            # Train transfer layer
            recon_target_proto = self.transfer_layer(self.source_model.proto_layer.prototypes)
            self.loss_transfer_layer = self.loss_recon(recon_target_proto, self.target_model.proto_layer.prototypes)
            print(f'transfer layer {self.epoch}: {self.loss_transfer_layer}')
            self.loss_transfer_layer.backward()

            self.optim_transfer_layer.step()
            self.optim_transfer_layer.zero_grad()


            self.epoch += 1

    def loss_pred(self, prediction, label):
        pred_loss = nn.CrossEntropyLoss()(prediction, label).mean()
        
        prediction_class = prediction.argmax(1)
        num_correct = (prediction_class == label).sum().item()
        class_accuracy = num_correct / len(prediction)

        return pred_loss, class_accuracy

    def loss_recon(self, input_, recon_image):
        return F.mse_loss(input_, recon_image).mean()

    def evaluate(self, target_test_dl):
        """ Evaluates the classification loss for the target test dataset

        """
        self.eval()
        batch_results = None
        with torch.no_grad():
            for xb, yb in target_test_dl:
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)

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
            }, path_name)

    @staticmethod
    def load_model(path_name, config, learning_rate):
        """
        Note:
            If used for inference, make sure to set model.eval()
        """
        print(f'Loading model from {path_name}')
        checkpoint = torch.load(path_name)

        loaded_model = TransferModel(config, learning_rate)

        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.epoch = checkpoint['epoch']

        return loaded_model