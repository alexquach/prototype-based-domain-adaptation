import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoLayer(nn.Module):
    def __init__(self, num_prototypes, latent_dim, **kwargs):
        super(ProtoLayer, self).__init__(**kwargs)
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.prototype_layer = nn.Linear(latent_dim, num_prototypes)

    def forward(self, latent_vectors):
        # TODO: Check if this is correct, or how to do uniform initialization
        self.prototypes = self.prototype_layer(latent_vectors)

        features_squared = ProtoLayer.get_norms(latent_vectors).view(-1, 1)
        protos_squared = ProtoLayer.get_norms(self.prototypes).view(1, -1)
        print("prototype: " + str(self.prototypes.size()))
        print("latent: " + str(latent_vectors.size()))
        dists_to_protos = features_squared + protos_squared - 2 * torch.dot(latent_vectors, torch.transpose(self.prototypes, 0, 1))

        alt_features_squared = ProtoLayer.get_norms(latent_vectors).view(1, -1)
        alt_protos_squared = ProtoLayer.get_norms(self.prototypes).view(-1, 1)
        dists_to_latents = alt_features_squared + alt_protos_squared - 2 * K.dot(self.prototypes, K.transpose(latent_vectors, 0, 1))
        
        return [dists_to_protos, dists_to_latents]
    
    @staticmethod
    def get_norms(x):
        return torch.sum(torch.pow(x, 2), dim=1)
