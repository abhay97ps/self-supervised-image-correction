import torch
import torchvision
from torchvision import transforms
import sys
from logger import Logger
from train import train
from pretext import GrayScalePL, SuperResolutionPL, RandomPatchPL, RealImagePL

# options
batch_size = 100
editor_lr = 0.005
discriminator_lr = 0.0005
num_of_epochs = 200
l_rec = 0.999
l_adv = 0.001
logger = Logger(model_name='CENC', data_name='STL-10', experiment_num='exp6')

# load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.447, 0.440, 0.407], [0.260, 0.256, 0.271])
])
data_dir = 'data/self_supervised/'
dataset = torchvision.datasets.ImageFolder(data_dir, transform)

# define tasks to work on
tasks = [
    RealImagePL,
    GrayScalePL,
    SuperResolutionPL,
    RandomPatchPL
]

# run experiment
train(dataset, tasks, batch_size, editor_lr,
      discriminator_lr, num_of_epochs, l_rec, l_adv, logger)
