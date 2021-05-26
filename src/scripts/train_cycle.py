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
    "num_prototypes": 20,
    "num_classes": 10,
    "use_convolution": False
}

proto_model_config_2 = {
    "input_dim": 784,
    "hidden_layers": [128, 128],
    "hidden_activations": [nn.ReLU(), None],
    "latent_dim": 128,
    "recon_activation": nn.Sigmoid(),
    "num_prototypes": 20,
    "num_classes": 10,
    "use_convolution": True
}

# Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# torch.autograd.set_detect_anomaly(True)

def train(model_name, config_1=proto_model_config_1, config_2=proto_model_config_2, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,\
          train_new=True, save_model=True, weights=(1,1,1,1,1,1,.1,.1,1,1), train_frac=1, nonlinear_transition=False,\
          load_source_model=None, load_target_model=None, freeze_source=False, t_recon_decay_weight=1, t_recon_decay_epochs = 10,\
          visualize_epochs=10, pretrain_proto_steps=1000, proto_align_iter_per_step=100):
    """ Trains a pair of Prototype Models and the mapping function between their latent spaces for domain adaptation
    Params:
        model_name: name used for saving/loading this cycle model
        config_1 (dict): ProtoModel configuration for source model
        config_2 (dict): ProtoModel configuration for target model
        epochs: epoch number to stop training at (note: this number is also the stopping point for existing models trained for nonzero epoch numbers)
        batch_size: batch_size
        train_new (bool): Whether to use an existing model or train a new one
        save_model (bool): Whether to save the result
        weights: weights for the custom loss function
        train_frac: fraction of the training dataset to use
        nonlinear_transition (bool): Whether to use a linear or nonlinear transition mapping 
        load_source_model (str): .pth file to load source model with
        load_target_model (str): .pth file to load target model with
        freeze_source (bool): Whether to allow gradient updates to the source model 
        t_recon_decay_weight: 
        t_recon_decay_epochs: 
        visualize_epochs (int): Visualizes the prototypes and samples every X training epochs
        pretrain_proto_steps: Number of iterations to repeat prototype alignment before the main training loop
        proto_align_iter_per_step: Number of iterations to repeat prototype alignment loss gradient per epoch
    """

    # load MNIST data
    mnist_train_dl, mnist_test_dl = load_mnist.load_mnist_dataloader(batch_size)
    svhn_train_dl, svhn_test_dl = load_svhn.load_svhn_dataloader(batch_size, greyscale=True, training_fraction=train_frac)

    # create Source ProtoModel
    if load_source_model:
        model_1 = ProtoModel.load_model(f"{load_source_model}", config_1, LEARNING_RATE)
    else:
        model_1 = ProtoModel(config_1, LEARNING_RATE)

    # create Source ProtoModel
    if load_target_model:
        model_2 = ProtoModel.load_model(f"{load_target_model}", config_2, LEARNING_RATE)
    else:
        model_2 = ProtoModel(config_2, LEARNING_RATE)

    cm = CycleModel(model_1, model_2, epochs=epochs, weights=weights, nonlinear_transition=nonlinear_transition, freeze_source=freeze_source, t_recon_decay_weight=t_recon_decay_weight, t_recon_decay_epochs=t_recon_decay_epochs)

    if train_new:
        cm.fit_combined_loss(mnist_train_dl, svhn_train_dl, visualize_epochs, model_name, pretrain_proto_steps=pretrain_proto_steps, proto_align_iter_per_step=proto_align_iter_per_step)
    else:
        cm = CycleModel.load_model(f"{model_name}.pth", model_1, model_2, epochs=epochs)
        cm.fit_combined_loss(mnist_train_dl, svhn_train_dl, visualize_epochs, model_name, pretrain_proto_steps=0, proto_align_iter_per_step=proto_align_iter_per_step)

    res = cm.evaluate(mnist_test_dl, lambda x: cm.source_model(x)[2])
    print("mnist: ", res)
    res = cm.evaluate(svhn_test_dl)
    print("svhn: ", res)
    res = cm.evaluate(mnist_test_dl, cm.predict_cross_domain_from_source)
    print("mnist -> svhn: ", res)
    res = cm.evaluate(svhn_test_dl, cm.predict_cross_domain_from_target)
    print("svhn -> mnist: ", res)

    cm.visualize_prototypes(f"{model_name}_proto.jpg")
    cm.visualize_samples(mnist_train_dl, svhn_train_dl, f"{model_name}_sample.jpg")
    cm.visualize_latent_2d(mnist_train_dl, svhn_train_dl, root_savepath=model_name, batch_multiple=5)
    print(f"weights: {weights} {train_frac} {'nonlinear' if nonlinear_transition else 'linear'}")

    if save_model:
        cm.save_model(f"{model_name}.pth")


if __name__ == "__main__": 
    #load_source_model="mnist_linear_1.pth", load_target_model="svhn_conv.pth"
    train("cm_class_both", train_new=True, train_frac=1, nonlinear_transition=True, freeze_source=True, t_recon_decay_weight=20, t_recon_decay_epochs=10)