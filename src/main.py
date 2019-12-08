import sys
from logger import Logger
from train import train
from data import load_unlabelled_data

# options
batch_size = 100
editor_lr = 0.0002
discriminator_lr = 0.0002
num_of_epochs = 100
l_rec = 0.999
l_adv = 0.001
logger = Logger(model_name='CENC', data_name='STL-10', experiment_num='exp1')

# load data
dataset = load_unlabelled_data()
# run experiment
train(dataset, batch_size, editor_lr, discriminator_lr,
      num_of_epochs, l_rec, l_adv, logger)
