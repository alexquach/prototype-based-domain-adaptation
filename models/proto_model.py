import torch.nn as nn
import torch.nn.functional as F

class ProtoModel(nn.Module):
    def __init__(self, num_prototypes, latent_dim, predictor_depth):
        super(ProtoModel, self).__init__()

        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.predictor_depth = predictor_depth

        self.classification_weight = 10
        self.reconstruction_weight = 1
        self.close_to_proto_weight = 1
        self.proto_close_to_weight = 1
        self.input_dim = 784

        self.encoder_layer1 = nn.Linear(self.input_dim, 128)
        self.encoder_layer2 = nn.Linear(128, 128)
        self.encoder_layer3 = nn.Linear(128, self.latent_dim)

        self.decoder_layer2 = nn.Linear(self.latent_dim, 128)
        self.decoder_layer1 = nn.Linear(128, 128)
        self.recons_layer = nn.Linear(128, self.input_dim)

        self.proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        self.encoder, self.decoder, self.predictor, self.proto_layer, self.latent, self.recons = self.build_parts()

        self.auto = self.build_model()

    def forward(self, input):
        xb = input.view(-1, self.input_dim)
        xb = self.encoder_layer1(xb)
        xb = F.relu(xb)
        xb = self.encoder_layer2(xb)
        xb = self.encoder_layer3(xb)

        proto_distances, feature_distances = self.proto_layer(xb)
        min_proto_dist = get_min(proto_distances)
        min_feature_dist = get_min(feature_distances)

        prediction = self.predictor.model(proto_distances)

        xb = self.decoder_layer2(xb)
        xb = self.decoder_layer1(xb)
        xb = F.relu(xb)
        xb = self.recons_layer(xb)
        recon = F.sigmoid(xb)

        return input, recon, prediction, min_proto_dist, min_feature_dist

    def loss(self, input, recon, prediction, min_proto_dist, min_feature_dist, label):

        recons_loss = mse(input, xb)
        pred_loss = categorical_crossentropy(label, prediction)
        proto_dist_loss = min_proto_dist.mean()
        feature_dist_loss = min_feature_dist.mean()

        overall_loss = self.reconstruction_weight * recons_loss.mean() +\
                       self.classification_weight * pred_loss.mean() +\
                       self.proto_close_to_weight * proto_dist_loss +\
                       self.close_to_proto_weight * feature_dist_loss
        
        return overall_loss
        