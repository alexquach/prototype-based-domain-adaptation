import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from models.proto_model import ProtoModel
from utils.plotting import plot_rows_of_images, plot_latent_tsne, plot_latent_pca

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class CycleModel(nn.Module):
    def __init__(self, source_model, target_model, epochs=10, weights=(1,1,1,1,1,1,.1,.1,1,1),\
                 nonlinear_transition=False, freeze_source=False, t_recon_decay_weight=1, t_recon_decay_epochs=0):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.source_model = source_model
        self.target_model = target_model
        self.nonlinear_transition = nonlinear_transition
        self.epoch = 0
        self.epochs = epochs
        self.t_recon_decay_weight = t_recon_decay_weight
        self.t_recon_decay_epochs = t_recon_decay_epochs

        # Transfer layer
        if nonlinear_transition:
            self.transition_model = nn.Sequential(
                nn.Linear(source_model.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, target_model.latent_dim)
            )

            self.inverse_transition_model = nn.Sequential(
                nn.Linear(target_model.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, source_model.latent_dim)
            )
        else:
            self.transition_model = nn.Linear(source_model.latent_dim, target_model.latent_dim)
            self.inverse_transition_model = Lambda(lambda x: (x - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T)))

        self.weight_recon_source, self.weight_recon_target, self.weight_autoencode_source,\
            self.weight_autoencode_target, self.weight_class_source, self.weight_class_target,\
            self.proto_close_to_weight, self.close_to_proto_weight, self.weight_class_transition,\
            self.weight_proto_align = weights

        combined_optim_params = [
            *self.target_model.parameters(),
            *self.transition_model.parameters(),
            *self.inverse_transition_model.parameters()
        ]

        if not freeze_source:
            combined_optim_params.extend(self.source_model.parameters())

        self.optim = optim.Adam(combined_optim_params)

        self.optim_transition = optim.Adam([
            self.target_model.proto_layer.prototypes,
            *self.transition_model.parameters(),
            *self.inverse_transition_model.parameters(),
        ])

        self.to(self.dev)

    def forward_base(self, xb_1, model_1, model_2, transition, inverse_transition):
        """ 

        Args:
            xb_1: input batch
            model_1: source model
            model_2: target model
            transition: transition layer between S -> T
            inverse_transition: transition layer between T -> S

        Returns:
            1. xb_1: The input batch
            2. transfer_recon_1: S -> T -> S reconstruction
            3. prediction_1: straight prediction from source model
            4. min_proto_dist_1: minimum prototype distance for batch
            5. min_feature_dist_1: min feature distance for batch
            6. transfer_recon_2: S -> T reconstructino
        """
        latent_1 = model_1.encoder(xb_1)
        latent_2 = transition(latent_1)

        # Get proto/feature distances
        proto_distances_1, feature_distances_1 = model_1.proto_layer(latent_1)
        min_proto_dist_1 = ProtoModel.get_min(proto_distances_1)
        min_feature_dist_1 = ProtoModel.get_min(feature_distances_1)
        prediction_1 = model_1.predictor(proto_distances_1)

        transfer_recon_2 = model_2.decoder(latent_2)
        transfer_latent_2 = model_2.encoder(transfer_recon_2)

        # inverse linear transfer (target -> source)
        #transfer_latent_source = (transfer_latent_target - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T))
        transfer_latent_1 = inverse_transition(transfer_latent_2)
        transfer_recon_1 = model_1.decoder(transfer_latent_1)

        return xb_1, transfer_recon_1, prediction_1, min_proto_dist_1, min_feature_dist_1, transfer_recon_2

    def forward_source(self, xb_source):
        return self.forward_base(xb_source, self.source_model, self.target_model, self.transition_model, self.inverse_transition_model)

    def forward_target(self, xb_target):
        return self.forward_base(xb_target, self.target_model, self.source_model, self.inverse_transition_model, self.transition_model)

    def predict_cross_domain_base(self, xb, model_1, model_2, transition):
        """ Takes in source data and converts it to target prediction via transition model + predictor """
        latent_1 = model_1.encoder(xb)
        latent_2 = transition(latent_1)
        transfer_recon_2 = model_2.decoder(latent_2)
        transfer_latent_2 = model_2.encoder(transfer_recon_2)
        proto_dist_2, _ = model_2.proto_layer(transfer_latent_2)
        prediction = model_2.predictor(proto_dist_2)

        return prediction

    def predict_cross_domain_from_source(self, xb_source):
        return self.predict_cross_domain_base(xb_source, self.source_model, self.target_model, self.transition_model)

    def predict_cross_domain_from_target(self, xb_target):
        return self.predict_cross_domain_base(xb_target, self.target_model, self.source_model, self.inverse_transition_model)


    def autoencode(self, xb_source, xb_target):
        latent_source = self.source_model.encoder(xb_source)
        recon_source = self.source_model.decoder(latent_source)

        latent_target = self.target_model.encoder(xb_target)
        recon_target = self.target_model.decoder(latent_target)

        return xb_source, recon_source, xb_target, recon_target

    def fit_combined_loss(self, source_train_dl, target_train_dl, visualize_10_epochs=False, model_name=None):
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
                _, recon_source, prediction_source, min_proto_dist_source, min_feature_dist_source, _ = self.forward_source(xb_source)
                _, recon_target, prediction_target, min_proto_dist_target, min_feature_dist_target, _ = self.forward_target(xb_target)

                # 1. Loss on transfered source
                loss_recon_source = self.loss_recon(xb_source, recon_source)

                # 2. Loss on transfered target
                loss_recon_target = self.loss_recon(xb_target, recon_target)

                # 3 + 4. Loss on autoencoded 
                _, autoencode_source, _, autoencode_target = self.autoencode(xb_source, xb_target) 
                loss_autoencode_source = self.loss_recon(xb_source, autoencode_source)
                loss_autoencode_target = self.loss_recon(xb_target, autoencode_target)

                # 5 + 6. Loss on straight-through classification error
                loss_class_source, acc_source = self.loss_pred(prediction_source, yb_source)
                loss_class_target, acc_target = self.loss_pred(prediction_target, yb_target)

                # 7 + 8. Loss on distances between prototypes -> samples and samples -> prototypes
                loss_proto_dist_source = torch.mean(min_proto_dist_source)
                loss_feature_dist_source = torch.mean(min_feature_dist_source)
                loss_proto_dist_target = torch.mean(min_proto_dist_target)
                loss_feature_dist_target = torch.mean(min_feature_dist_target)

                if self.epoch < self.t_recon_decay_epochs:
                    weight_recon_target_adj = self.weight_recon_target + (float)(self.t_recon_decay_epochs - self.epochs)/self.t_recon_decay_epochs *self.t_recon_decay_weight
                else:
                    weight_recon_target_adj = self.weight_recon_target

                # calculate combined loss
                self.loss_combined = self.weight_recon_source * loss_recon_source +\
                                     weight_recon_target_adj * loss_recon_target +\
                                     self.weight_autoencode_source * loss_autoencode_source +\
                                     self.weight_autoencode_target * loss_autoencode_target +\
                                     self.weight_class_source * loss_class_source +\
                                     self.weight_class_target * loss_class_target +\
                                     self.proto_close_to_weight * (loss_proto_dist_source + loss_proto_dist_target) +\
                                     self.close_to_proto_weight * (loss_feature_dist_source + loss_feature_dist_target)
                self.loss_combined.backward(retain_graph=True)

                self.optim.step()
                self.optim.zero_grad()
                
                # 9. Loss on fake data from transitioning source training data to target domain
                prediction_transition = self.predict_cross_domain_from_source(xb_source)
                loss_class_transition, acc_transition = self.loss_pred(prediction_transition, yb_source)

                # 10. Loss on prototype alignment
                # loss_proto_align = self.loss_recon(self.source_model.proto_layer.prototypes, self.target_model.proto_layer.prototypes)
                loss_proto_align = self.loss_recon(self.source_model.proto_layer.prototypes, self.inverse_transition_model(self.target_model.proto_layer.prototypes))+\
                                   self.loss_recon(self.target_model.proto_layer.prototypes, self.transition_model(self.source_model.proto_layer.prototypes))

                loss_transition = self.weight_class_transition * loss_class_transition +\
                                  self.weight_proto_align * loss_proto_align
                loss_transition.backward()

                self.optim_transition.step()
                self.optim_transition.zero_grad()

            print(f'\ntransfer source {self.epoch}: {loss_recon_source}')
            print(f'transfer target {self.epoch}: {loss_recon_target}')
            print(f'autoencode {self.epoch}: {loss_autoencode_source} + {loss_autoencode_target}')
            print(f'class loss {self.epoch}: {loss_class_source} + {loss_class_target}')
            print(f'class acc {self.epoch}: {acc_source} + {acc_target}')
            print(f'proto dist {self.epoch}: {loss_proto_dist_source} + {loss_proto_dist_target}')
            print(f'feat dist {self.epoch}: {loss_feature_dist_source} + {loss_feature_dist_target}')
            print(f'transition loss/acc {self.epoch}: {loss_class_transition} + {acc_transition}')
            print(f'prototype alignment loss {self.epoch}: {loss_proto_align}')
            self.epoch += 1

            if visualize_10_epochs and (self.epoch % 10 == 0):
                if model_name:
                    self.visualize_prototypes(f"{model_name}_proto_e{self.epoch}.jpg")
                    self.visualize_samples(source_train_dl, target_train_dl, f"{model_name}_sample_e{self.epoch}.jpg")
                    self.visualize_latent_2d(source_train_dl, target_train_dl, root_savepath=f"{model_name}_e{self.epoch}", batch_multiple=5)
                else:
                    self.visualize_prototypes()
                    self.visualize_samples(source_train_dl, target_train_dl)
                    self.visualize_latent_2d(source_train_dl, target_train_dl, batch_multiple=5)

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
                    prediction = eval_model(xb)
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
                            self.weight_autoencode_target, self.weight_class_source, self.weight_class_target,\
                            self.proto_close_to_weight, self.close_to_proto_weight, self.weight_class_transition,\
                            self.weight_proto_align)
            }, path_name)

    @staticmethod
    def load_model(path_name, model_1, model_2, epochs=10):
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

        loaded_model.epochs = epochs

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

        _, recon_transfer_source, _, _, _, recon_transfer_intermediate_source = self.forward_source(xb_source)
        _, recon_transfer_target, _, _, _, recon_transfer_intermediate_target = self.forward_target(xb_target)

        plot_rows_of_images([xb_source, xb_target, recon_source, recon_target, recon_transfer_source, recon_transfer_target], recon_transfer_intermediate_source, recon_transfer_intermediate_target, path_name)

    def visualize_latent_2d(self, source_dl, target_dl, root_savepath=None, batch_multiple=1):
        """ 
        Note:
            Savepath should not include the extension (.jpg, .png)

        Plots the following:
            1. Source Prototypes
            2. Target Prototypes
            3. Transition Target (Source -> Target) Prototypes
            4. Transition Source (Target -> Source) Prototypes
            5. Recon Source (S -> T -> S) Prototypes
            6. Recon Target (T -> S -> T) Prototypes
            7. 
        """
        proto_mult = int(self.source_model.num_prototypes / self.source_model.num_classes)

        # prototypes
        source = self.source_model.proto_layer.prototypes
        target = self.target_model.proto_layer.prototypes
        transition_target = self.transition_model(source)
        transition_source = self.inverse_transition_model(target)
        recon_source = self.inverse_transition_model(transition_target)
        recon_target = self.transition_model(transition_source)

        # samples
        xb_source, yb_source = next(iter(source_dl))
        xb_target, yb_target = next(iter(target_dl))
        # build up to batch_multiple * batch_size number of samples
        for i in range(batch_multiple - 1):
            new_xb_source, new_yb_source = next(iter(source_dl))
            new_xb_target, new_yb_target = next(iter(target_dl))
            xb_source = torch.cat([xb_source, new_xb_source])
            yb_source = torch.cat([yb_source, new_yb_source])
            xb_target = torch.cat([xb_target, new_xb_target])
            yb_target = torch.cat([yb_target, new_yb_target])
        xb_source = self.source_model.encoder(xb_source.to(self.dev))
        xb_target = self.target_model.encoder(xb_target.to(self.dev))
        yb_source = yb_source.to(self.dev).cpu().detach().numpy()
        yb_target = yb_target.to(self.dev).cpu().detach().numpy()
        xb_transition_target = self.transition_model(xb_source)
        xb_transition_source = self.inverse_transition_model(xb_target)
        xb_recon_source = self.inverse_transition_model(xb_transition_target)
        xb_recon_target = self.transition_model(xb_transition_source)

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, 2)

        proto_to_plot = [source, target, transition_target, transition_source, recon_source, recon_target]
        sample_to_plot = [xb_source, xb_target, xb_transition_target, xb_transition_source, xb_recon_source, xb_recon_target]
        sample_labels = [yb_source, yb_target, yb_source, yb_target, yb_source, yb_target]

        for i in range(3):
            for j in range(2):
                new_ax = plt.subplot(gs[i, j])

                #plot_latent_pca(proto_to_plot[i*2 +j], range(10), ax=new_ax, marker='x')
                plot_latent_tsne([proto_to_plot[i*2 +j], sample_to_plot[i*2 + j]], [list(range(10)) * proto_mult, sample_labels[i*2 +j]], ax=new_ax, markers=['x', 'o'], sizes=[300, 1], fig=fig)

        # plots proto + samples for: Source, transition source
        new_ax = plt.subplot(gs[3, 0])
        plot_latent_tsne([source, xb_source, transition_source, xb_transition_source], [list(range(10)) * proto_mult, yb_source, list(range(10)) * proto_mult, yb_source], ax=new_ax, markers=['x', '|', '+', '_'], sizes=[300, 5, 300, 5], fig=fig)

        # plots proto + samples for: Target, transition target
        new_ax = plt.subplot(gs[3, 1])
        plot_latent_tsne([target, xb_target, transition_target, xb_transition_target], [list(range(10)) * proto_mult, yb_target, list(range(10)) * proto_mult, yb_target], ax=new_ax, markers=['x', '|', '+', '_'], sizes=[300, 5, 300, 5], fig=fig)

        plt.show()
        if root_savepath:
            plt.savefig(root_savepath + "_latent_2d.jpg")

        # plot_latent(source, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_source.jpg")
        # plot_latent(target, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_target.jpg")
        # plot_latent(transition_target, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_transition_target.jpg")
        # plot_latent(transition_source, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_transition_source.jpg")
        # plot_latent(recon_source, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_recon_source.jpg")
        # plot_latent(recon_target, range(10), savepath=root_savepath if root_savepath is None else root_savepath + "_recon_target.jpg")