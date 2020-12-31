""" Basic script for training and testing a single ProtoModel """
from data_parsing.load_mnist import load_mnist_dataloader
from models.proto_model import ProtoModel

# Hyperparameters
NUM_PROTOS = 10
HIDDEN1_DIM = 128
HIDDEN2_DIM = 128
LATENT_DIM = 128
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# load MNIST data
train_dl, valid_dl = load_mnist_dataloader(BATCH_SIZE)

# create ProtoModel
model = ProtoModel(NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, LEARNING_RATE)

model.fit(NUM_EPOCHS, train_dl, valid_dl)