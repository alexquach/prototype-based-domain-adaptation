""" Basic script for training and testing two ProtoModels and comparing their prototypes """
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_parsing import load_mnist, load_svhn
from models.proto_model import ProtoModel
from models.latent_transition import LatentTransition

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
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def train(model_name_1, model_name_2, epochs=NUM_EPOCHS, train_new=True, save_model=True, weights=(1, 1, 1, 1)):

    # load MNIST data
    mnist_train_dl, mnist_test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)
    svhn_train_dl, svhn_test_dl = load_svhn.load_svhn_dataloader(BATCH_SIZE, greyscale=True)

    # create ProtoModel
    model_1 = ProtoModel(proto_model_config_1, LEARNING_RATE)
    model_2 = ProtoModel(proto_model_config_2, LEARNING_RATE)

    if train_new:
        model_1.fit(NUM_EPOCHS, mnist_train_dl, visualize_sample_name=None)
        model_2.fit(NUM_EPOCHS, svhn_train_dl, visualize_sample_name=None)
    else:
        model_1 = ProtoModel.load_model(f"{model_name_1}.pth", proto_model_config_1, LEARNING_RATE)
        model_2 = ProtoModel.load_model(f"{model_name_2}.pth", proto_model_config_2, LEARNING_RATE)

    # generate test loss metrics
    evaluate_models = False
    if evaluate_models:
        train_losses = model_1.evaluate(mnist_train_dl)
        print(train_losses)
        test_losses = model_1.evaluate(mnist_test_dl)
        print(test_losses)

        train_losses = model_2.evaluate(svhn_train_dl)
        print(train_losses)
        test_losses = model_2.evaluate(svhn_test_dl)
        print(test_losses)

    model_1.visualize_prototypes(path_name=f"{model_name_1}_proto.jpg")
    model_2.visualize_prototypes(path_name=f"{model_name_2}_proto.jpg")
    # model_2.visualize_sample(mnist_test_dl, path_name="svhn_mnist_sample.jpg")
    # model_2.visualize_sample(svhn_test_dl, path_name="svhn_svhn_sample.jpg")

    lt = LatentTransition(model_1, model_2, epochs=NUM_EPOCHS)
    lt.fit()

    lt.visualize_transformed_source_prototype("mnist_svhn_proto.jpg")
    lt.visualize_latent(model_1.proto_layer.prototypes, range(10), f"{model_name_1}_latent.jpg")
    lt.visualize_latent(model_2.proto_layer.prototypes, range(10), f"{model_name_1}_latent.jpg")
    lt.visualize_latent(lt.transition_model(model_1.proto_layer.prototypes), range(10), f"{model_name_1}_transfer_latent_nonlinear.jpg")
    # lt.visualize_sample(mnist_test_dl, "mnist_svhn_sample.jpg")
    eval_ = lt.evaluate(mnist_test_dl)
    print(eval_)

    # Save model
    if save_model:
        model_1.save_model(f"{model_name_1}.pth")
        model_2.save_model(f"{model_name_2}.pth")

if __name__ == "__main__": 
    train("mnist_linear", "svhn_conv", train_new=False, save_model=True)