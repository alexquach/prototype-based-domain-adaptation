import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from models.proto_layer import ProtoLayer
from models.predictor import Predictor

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)

class ProtoModel(nn.Module):
    def __init__(self, num_prototypes, hidden1_dim, hidden2_dim, latent_dim, num_classes, learning_rate):
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

        self.build_parts()

        self.optim = optim.SGD(self.parameters(), lr=learning_rate)

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

    @staticmethod
    def get_min(x):
        return torch.min(x, dim=1).values

    def forward(self, input_):
        latent = self.encoder(input_)
        
        proto_distances, feature_distances = self.proto_layer(latent)
        min_proto_dist = self.get_min(proto_distances)
        min_feature_dist = self.get_min(feature_distances)
        
        recons = self.decoder(latent)

        prediction = self.predictor(proto_distances)

        return input_, recons, prediction, min_proto_dist, min_feature_dist

    def loss(self, input_, recons, prediction, min_proto_dist, min_feature_dist, label):

        recons_loss = F.mse_loss(input_, recons)
        nn.CrossEntropyLoss()
        #pred_loss_1 = nn.NLLLoss()(torch.log(nn.Softmax(dim=1)(prediction)), label)
        pred_loss = nn.CrossEntropyLoss()(prediction, label)
        proto_dist_loss = torch.mean(min_proto_dist)
        feature_dist_loss = torch.mean(min_feature_dist)

        overall_loss = self.reconstruction_weight * recons_loss.mean() +\
                       self.classification_weight * pred_loss.mean() +\
                       self.proto_close_to_weight * proto_dist_loss +\
                       self.close_to_proto_weight * feature_dist_loss
        
        return overall_loss
        
    def fit(self, epochs, train_dl, valid_dl):
        for epoch in range(epochs):
            self.train()
            for xb, yb in train_dl:
                input_, recons, prediction, min_proto_dist, min_feature_dist = self.__call__(xb)
                
                loss = self.loss(input_, recons, prediction, min_proto_dist, min_feature_dist, yb)
                print(f'{epoch} + {loss}')
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()

