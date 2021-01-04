""" Basic script for training and testing a single ProtoModel """
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
LEARNING_RATE = 0.0001

# load MNIST data
train_dl, valid_dl = load_mnist.load_mnist_dataloader(BATCH_SIZE)

# create ProtoModel
model = ProtoModel(NUM_PROTOS, HIDDEN1_DIM, HIDDEN2_DIM, LATENT_DIM, NUM_CLASSES, LEARNING_RATE, use_convolution=True)

model.fit(NUM_EPOCHS, train_dl, valid_dl)