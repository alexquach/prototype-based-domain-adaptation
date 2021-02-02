""" Basic script for training and testing two ProtoModels configured in a CycleModel """
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_parsing import load_mnist, load_svhn
from models.proto_model import ProtoModel
from models.cycle_model import CycleModel

proto_model_config_1 = {
    "input_dim": 784,
    "hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
    "latent_dim": 128,
    "recon_activation": nn.Sigmoid(),
    "num_prototypes": 10,
    "num_classes": 10,
    "use_convolution": False
}

proto_model_config_2 = {
    "input_dim": 784,
    "hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
    "latent_dim": 128,
    "recon_activation": nn.Sigmoid(),
    "num_prototypes": 10,
    "num_classes": 10,
    "use_convolution": True
}

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# torch.autograd.set_detect_anomaly(True)

def train(model_name, config_1=proto_model_config_1, config_2=proto_model_config_2, epochs=NUM_EPOCHS,\
          train_new=True, save_model=True, weights=(1,1,1,1,1,1,.1,.1,1), train_frac=1, nonlinear_transition=False,\
          load_source_model=None, freeze_source=False):

    # load MNIST data
    mnist_train_dl, mnist_test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)
    svhn_train_dl, svhn_test_dl = load_svhn.load_svhn_dataloader(BATCH_SIZE, greyscale=True, training_fraction=train_frac)

    # create ProtoModel
    if load_source_model:
        model_1 = ProtoModel.load_model(f"{load_source_model}", proto_model_config_1, LEARNING_RATE)
    else:
        model_1 = ProtoModel(config_1, LEARNING_RATE)
    model_2 = ProtoModel(config_2, LEARNING_RATE)

    cm = CycleModel(model_1, model_2, epochs=epochs, weights=weights, nonlinear_transition=nonlinear_transition, freeze_source=freeze_source)

    if train_new:
        cm.fit_combined_loss(mnist_train_dl, svhn_train_dl)
    else:
        cm = CycleModel.load_model(f"{model_name}.pth", model_1, model_2)

    res = cm.evaluate(svhn_test_dl)
    print(res)
    res = cm.evaluate(mnist_test_dl, cm.source_model)
    print(res)

    cm.visualize_prototypes(f"{model_name}_proto.jpg")
    cm.visualize_samples(mnist_train_dl, svhn_train_dl, f"{model_name}_sample.jpg")
    print(f"weights: {weights} {train_frac} {'nonlinear' if nonlinear_transition else 'linear'}")

    if save_model:
        cm.save_model(f"{model_name}.pth")


if __name__ == "__main__": 
    train("cm_class_both", train_new=True, train_frac=1, nonlinear_transition=True, load_source_model="mnist_linear_1.pth", freeze_source=True)