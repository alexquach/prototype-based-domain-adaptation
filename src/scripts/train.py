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

    "mnist_conv": True,
    "use_convolution": True,
    "conv_input_channels": 1,
    "conv_hidden_layers": [128, 128],
    "conv_hidden_activations": [nn.ReLU(), None],
}

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def train(model_name, config=proto_model_config, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, train_frac=1, override_conv=False, mnist_conv=False,\
    train_new=True, save_model=True, dataset="mnist"):
    """ Trains a single ProtoModel 
    
    Params:
        model_name: name used for saving/loading this cycle model
        config: configuration for the ProtoModel
        epochs: epoch number to stop training at (note: this number is also the stopping point for existing models trained for nonzero epoch numbers)
        batch_size: batch_size
        train_frac: fraction of training dataset to use
        override_conv: if true, overrides config to use_convoluition
        mnist_conv: whether to use convolutional version of mnist
        train_new (bool): Whether to use an existing model or train a new one
        save_model (bool): Whether to save the result
        dataset (str): dataset 'svhn' or 'mnist' to train model on

    """

    if mnist_conv:
        proto_model_config['mnist_conv'] = True
    elif override_conv:
        proto_model_config['use_convolution'] = True

    # load test data
    if dataset == "svhn":
        train_dl, test_dl = load_svhn.load_svhn_dataloader(batch_size, greyscale=True, training_fraction=train_frac)
    else:
        train_dl, test_dl = load_mnist.load_mnist_dataloader(batch_size)

    # create ProtoModel
    model = ProtoModel(proto_model_config, LEARNING_RATE)
    model.to(dev)

    if train_new:
        model.fit(epochs, train_dl, visualize_sample_name=None)
    else:
        model = ProtoModel.load_model(f"{model_name}.pth", proto_model_config, LEARNING_RATE)
        model.fit(epochs, train_dl, visualize_sample_name=None)

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
    train("mnist_full_conv_30", train_new=True, save_model=True, dataset="mnist", mnist_conv=True)