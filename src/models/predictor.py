import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Predictor(nn.Module):

    def __init__(self, num_prototypes, layers, num_classes):
        """ Takes in prototype distances and outputs predicted class

        Args:
            num_prototypes (int): The number of prototypes we will be predicting from
            layers (list): List of the number of nodes per layer
            num_classes (int): The number of classes to finally predict from
        """
        super(Predictor, self).__init__()
        
        self.num_prototypes = num_prototypes
        self.layers = layers
        self.num_classes = num_classes

        if layers:
            pass
            # TODO: generate hidden layers
            self.final_layer = nn.Linear(self.num_prototypes, self.num_classes)
        else:
            self.final_layer = nn.Sequential(
                Lambda(lambda x: torch.neg(x)),
                nn.Softmax()
            )

    def forward(self, proto_distances):
        return self.final_layer(proto_distances)

