import sys

# options
batch_size = 100
editor_lr = 0.0002
discriminator_lr = 0.0002
num_of_epochs = 100
l_rec = 0.999
l_adv = 0.001
logger = Logger(model_name='CENC', data_name='STL-10', experiment_num='exp1')
