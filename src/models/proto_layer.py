import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoLayer(nn.Module):
    def __init__(self, num_prototypes, latent_dim, **kwargs):
        super(ProtoLayer, self).__init__(**kwargs)
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.prototype_layer = nn.Linear(latent_dim, num_prototypes)

        # TODO: Restructure as Parameter and do uniform initialization
        # self.prototypes = nn.Parameter(data=Torch.Tensor(outputdim, inputdim, et))
        # nn.init.uniform_(self.prototype_layer.weight)
        # nn.init.uniform_(self.params, -0.5, 0.5)
        # nn.Parameter(data=torch.zeros(self.model_dim, self.rank))
        self.prototypes = self.prototype_layer.weight

    def forward(self, latent_vectors):
        """ Calculates L2 distances between encoded input (`latent_vectors`) and the prototypes.

        Note:
            Uses fancy matrix computation technique that is not immediately obvious in the paper
            https://github.com/OscarcarLi/PrototypeDL/blob/master/autoencoder_helpers.py
        """
        features_squared = ProtoLayer.get_norms(latent_vectors).view(-1, 1)
        protos_squared = ProtoLayer.get_norms(self.prototypes).view(1, -1)
        dists_to_protos = features_squared + protos_squared - 2 * torch.matmul(latent_vectors, torch.transpose(self.prototypes, 0, 1))

        alt_features_squared = ProtoLayer.get_norms(latent_vectors).view(1, -1)
        alt_protos_squared = ProtoLayer.get_norms(self.prototypes).view(-1, 1)
        dists_to_latents = alt_features_squared + alt_protos_squared - 2 * torch.matmul(self.prototypes, torch.transpose(latent_vectors, 0, 1))
        
        return [dists_to_protos, dists_to_latents]
    
    @staticmethod
    def get_norms(x):
        return torch.sum(torch.pow(x, 2), dim=1)
