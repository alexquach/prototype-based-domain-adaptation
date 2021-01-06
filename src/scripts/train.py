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
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# load MNIST data
train_dl, test_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)

# create ProtoModel
model = ProtoModel(NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=False)

#model.fit(NUM_EPOCHS, train_dl)
model = ProtoModel.load_model("testing.pth", NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=False)

# generate test loss metrics
overall_loss = model.evaluate(test_dl, overall_metric=True)
print(overall_loss)
losses = model.evaluate(test_dl, overall_metric=False)
print(losses)



# Testing the save/load functionality
test_save = False
if test_save:
    model.save_model("testing.pth")
    new_model = ProtoModel.load_model("testing.pth", NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=False)
    new_model.fit(10, train_dl)