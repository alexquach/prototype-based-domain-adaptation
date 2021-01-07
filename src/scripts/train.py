""" Basic script for training and testing a single ProtoModel """
import torch
from data_parsing import load_mnist
from models.proto_model import ProtoModel

# Hyperparameters
NUM_PROTOS = 12
HIDDEN1_DIM = 128
HIDDEN2_DIM = 128
LATENT_DIM = 128
NUM_CLASSES = 10
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# load MNIST data
train_dl, test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)

# create ProtoModel
model = ProtoModel(NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=True)

#model.fit(NUM_EPOCHS, train_dl, visualize_samples=True)
model = ProtoModel.load_model("testing_conv.pth", NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=True)

# generate test loss metrics
train_losses = model.evaluate(train_dl)
print(train_losses)
test_losses = model.evaluate(test_dl)
print(test_losses)

model.visualize_sample(test_dl, path_name="conv_test.jpg")
model.visualize_prototypes(path_name="conv_test_proto")

# Testing the save/load functionality
test_save = False
if test_save:
    model.save_model("testing_conv.pth")
    new_model = ProtoModel.load_model("testing_conv.pth", NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=True)
    #new_model.fit(10, train_dl)