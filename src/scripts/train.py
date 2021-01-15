""" Basic script for training and testing a single ProtoModel """
import torch
import torch.nn as nn
from data_parsing import load_mnist, load_svhn
from models.proto_model import ProtoModel

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

proto_model_config = {
    "input_dim": 784,
    "latent_dim": 128,
    "num_prototypes": 10,
    "num_classes": 10,

    "hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
    "recon_activation": nn.Sigmoid(),

    "use_convolution": True,
    "conv_input_channels": 1,
    "conv_hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
}

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# load test data
train_dl, test_dl = load_svhn.load_svhn_dataloader(BATCH_SIZE, greyscale=True)

# create ProtoModel
model = ProtoModel(proto_model_config, LEARNING_RATE)
model.to(dev)

train_new = True
if train_new:
    model.fit(NUM_EPOCHS, train_dl, visualize_sample_name="svhn_conv")
else:
    model = ProtoModel.load_model("svhn_conv.pth", proto_model_config, LEARNING_RATE)

# generate test loss metrics
train_losses = model.evaluate(train_dl)
print(train_losses)
test_losses = model.evaluate(test_dl)
print(test_losses)

model.visualize_sample(test_dl, path_name="svhn_conv_sample.jpg")
model.visualize_prototypes(path_name="svhn_conv_proto.jpg")

# Testing the save/load functionality
test_save = True
if test_save:
    model.save_model("svhn_conv.pth")
    new_model = ProtoModel.load_model("svhn_conv.pth", proto_model_config, LEARNING_RATE)
    #new_model.fit(10, train_dl)