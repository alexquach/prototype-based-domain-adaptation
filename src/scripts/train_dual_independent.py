""" Basic script for training and testing two ProtoModels and comparing their prototypes """
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_parsing import load_mnist, load_svhn
from models.proto_model import ProtoModel

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
    "use_convolution": False
}

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# load MNIST data
mnist_train_dl, mnist_test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)
svhn_train_dl, svhn_test_dl = load_svhn.load_svhn_dataloader(BATCH_SIZE, greyscale=True)

# create ProtoModel
model_1 = ProtoModel(proto_model_config_1, LEARNING_RATE)
model_2 = ProtoModel(proto_model_config_2, LEARNING_RATE)

train_new = False
if train_new:
    model_1.fit(NUM_EPOCHS, mnist_train_dl, visualize_sample_name=None)
    model_2.fit(NUM_EPOCHS, svhn_train_dl, visualize_sample_name=None)
else:
    model_1 = ProtoModel.load_model("mnist_1.pth", proto_model_config_1, LEARNING_RATE)
    model_2 = ProtoModel.load_model("svhn_2.pth", proto_model_config_2, LEARNING_RATE)

# generate test loss metrics
evaluate_models = True
if evaluate_models:
    train_losses = model_1.evaluate(mnist_train_dl)
    print(train_losses)
    test_losses = model_1.evaluate(mnist_test_dl)
    print(test_losses)

    train_losses = model_2.evaluate(svhn_train_dl)
    print(train_losses)
    test_losses = model_2.evaluate(svhn_test_dl)
    print(test_losses)

model_1.visualize_prototypes(path_name="mnist_proto_1")
model_2.visualize_prototypes(path_name="svhn_proto_2")
model_2.visualize_sample(mnist_test_dl, path_name="mnist_sample.jpg")

proto_1 = model_1.proto_layer.prototypes
proto_2 = model_2.proto_layer.prototypes

lt = model_1.generate_latent_transition(model_2, mnist_train_dl)

# lt.visualize_transformed_source_prototype("mnist_svhn_proto.jpg")
# lt.visualize_sample(mnist_test_dl, "mnist_svhn_sample.jpg")
eval_ = lt.evaluate(mnist_test_dl)
print(eval_)


# Testing the save/load functionality
test_save = False
if test_save:
    model_1.save_model("mnist_1.pth")
    model_2.save_model("svhn_2.pth")
    
    #new_model = ProtoModel.load_model("testing.pth", proto_model_config, LEARNING_RATE)
    #new_model.fit(10, train_dl)