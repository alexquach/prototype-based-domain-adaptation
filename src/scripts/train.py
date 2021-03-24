""" Basic script for training and testing a single ProtoModel """
import torch
import torch.nn as nn
from data_parsing import load_mnist, load_svhn
from models.proto_model import ProtoModel

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

proto_model_config = {
    "input_dim": 784,
    "latent_dim": 32,
    "num_prototypes": 30,
    "num_classes": 10,
    "proto_dropout": 0,

    "hidden_layers": [128, 128],
    "hidden_activations": [nn.Sigmoid(), None],
    "recon_activation": nn.Sigmoid(),

    "use_convolution": True,
    "conv_input_channels": 1,
    "conv_hidden_layers": [128, 128],
    "conv_hidden_activations": [nn.ReLU(), None],
}

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def train(model_name, config=proto_model_config, epochs=NUM_EPOCHS, override_conv=False,\
    train_new=True, save_model=True, dataset="mnist"):

    if override_conv:
        proto_model_config['use_convolution'] = True

    # load test data
    if dataset == "svhn":
        train_dl, test_dl = load_svhn.load_svhn_dataloader(BATCH_SIZE, greyscale=True)
    else:
        train_dl, test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)

    # create ProtoModel
    model = ProtoModel(proto_model_config, LEARNING_RATE)
    model.to(dev)

    if train_new:
        model.fit(NUM_EPOCHS, train_dl, visualize_sample_name=None)
    else:
        model = ProtoModel.load_model(f"{model_name}.pth", proto_model_config, LEARNING_RATE)

    # generate test loss metrics
    train_losses = model.evaluate(train_dl)
    print(train_losses)
    test_losses = model.evaluate(test_dl)
    print(test_losses)

    model.visualize_sample(test_dl, path_name=f"{model_name}_sample.jpg")
    model.visualize_prototypes(path_name=f"{model_name}_proto.jpg")

    # Testing the save/load functionality
    if save_model:
        model.save_model(f"{model_name}.pth")
        new_model = ProtoModel.load_model(f"{model_name}.pth", proto_model_config, LEARNING_RATE)
        #new_model.fit(10, train_dl)

if __name__ == "__main__": 
    train("svhn_conv_30", train_new=True, save_model=True, dataset="svhn")