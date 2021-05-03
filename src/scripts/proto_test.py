import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from models.proto_model import ProtoModel
from models.proto_layer import ProtoLayer
from utils.plotting import plot_rows_of_images, plot_latent_tsne, plot_latent_pca

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ProtoCycleModel(nn.Module):
    def __init__(self, epochs=10, weights=1, latent_dim=256, num_protos=30, num_classes=10,\
                 nonlinear_transition=False, freeze_source=False):
        super().__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.proto_layer_1 = ProtoLayer(num_protos, latent_dim)
        self.proto_layer_2 = ProtoLayer(num_protos, latent_dim)
        self.latent_dim = latent_dim
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.nonlinear_transition = nonlinear_transition
        self.epoch = 0
        self.epochs = epochs

        # Transfer layer
        if nonlinear_transition:
            self.transition_model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )

            self.inverse_transition_model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
        else:
            self.transition_model = nn.Linear(latent_dim, latent_dim)
            self.inverse_transition_model = Lambda(lambda x: (x - self.transition_model.bias).matmul(torch.inverse(self.transition_model.weight.T)))

        self.weight_proto_align = weights

        optim_transition_params = [
            *self.proto_layer_2.parameters(),
            *self.transition_model.parameters(),
            *self.inverse_transition_model.parameters()
        ]

        if not freeze_source:
            optim_transition_params.extend(self.proto_layer_1.parameters())

        self.optim_transition = optim.Adam(optim_transition_params)

        self.to(self.dev)


    def fit_combined_loss(self):
        """
        Trains using a combined loss for simultaneous optimization

        """

        while self.epoch < self.epochs:
            self.train()

            # 10. Loss on prototype alignment
            proto_losses_source, proto_losses_target = self.group_proto_loss()
            loss_proto_align = torch.mean(proto_losses_source) + torch.mean(proto_losses_target)

            loss_transition = self.weight_proto_align * loss_proto_align
            loss_transition.backward()

            self.optim_transition.step()
            self.optim_transition.zero_grad()

            if self.epoch % 100 == 0:
                print(f"proto loss: {proto_losses_source} and {proto_losses_target}")
                print(f"loss_transition: {loss_transition}")
                print(f'prototype loss (src/tgt) {self.epoch}: \n {proto_losses_source} \n {proto_losses_target}')
            self.epoch += 1

    def compute_pairwise_dist(self, input1, input2):
        source_squared = ProtoLayer.get_norms(input1).view(-1, 1)
        target_squared = ProtoLayer.get_norms(input2).view(1, -1)
        source_to_target = source_squared + target_squared - 2 * torch.matmul(input1, torch.transpose(input2, 0, 1))

        return source_to_target

    def group_proto_loss(self):
        """ each prototype in source/target domain is close to at least one corresponding class prototypes in other domain """
        num_protos = self.num_protos
        num_classes = self.num_classes

        proto_losses_source = torch.tensor([])
        proto_losses_target = torch.tensor([])

        for proto in range(num_classes):
            a = self.compute_pairwise_dist(self.proto_layer_1.prototypes[proto::num_classes], self.inverse_transition_model(self.proto_layer_2.prototypes[proto::num_classes])).to(self.dev)
            b = self.compute_pairwise_dist(self.proto_layer_2.prototypes[proto::num_classes], self.transition_model(self.proto_layer_1.prototypes[proto::num_classes])).to(self.dev)

            proto_losses_source = torch.cat([proto_losses_source, torch.mean(a.min(axis=1).values).reshape(1)])
            proto_losses_target = torch.cat([proto_losses_target, torch.mean(b.min(axis=1).values).reshape(1)])

        # use diff loss
        return proto_losses_source, proto_losses_target

    def loss_pred(self, prediction, label):
        pred_loss = nn.CrossEntropyLoss()(prediction, label).mean()
        prediction_class = prediction.argmax(1)
        num_correct = (prediction_class == label).sum().item()
        class_accuracy = num_correct / len(prediction)

        return pred_loss, class_accuracy

    def loss_recon(self, input_, recon_image):
        return F.mse_loss(input_.view(input_.shape[0], -1), recon_image).mean()

def train():
    pcm = ProtoCycleModel(epochs=10000)

    pcm.fit_combined_loss()

if __name__ == "__main__": 
    train()