""" Basic script for training and testing a single ProtoModel """
import torch
import torch.nn as nn
from data_parsing import load_mnist
from models.proto_model import ProtoModel

proto_model_config = {
    "input_dim": 784,
    "hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
    "latent_dim": 128,
    "recon_activation": nn.Sigmoid(),
    "num_prototypes": 12,
    "num_classes": 10,
    "use_convolution": False
}

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# load MNIST data
train_dl, test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)

# create ProtoModel
model = ProtoModel(proto_model_config, LEARNING_RATE)

model.fit(NUM_EPOCHS, train_dl, visualize_samples=True)
#model = ProtoModel.load_model("testing.pth", proto_model_config, INPUT_DIM, NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=False)

# generate test loss metrics
train_losses = model.evaluate(train_dl)
print(train_losses)
test_losses = model.evaluate(test_dl)
print(test_losses)

# model.visualize_sample(test_dl, path_name="conv_test.jpg")
# model.visualize_prototypes(path_name="conv_test_proto")

# Testing the save/load functionality
test_save = True
if test_save:
    model.save_model("testing.pth")
    new_model = ProtoModel.load_model("testing.pth", proto_model_config, LEARNING_RATE)
    #new_model.fit(10, train_dl)