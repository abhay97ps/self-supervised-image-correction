import torch
import torchvision
from torchvision import transforms
from editor import editor18
import matplotlib.pyplot as plt
from pretext import GrayScalePL, SuperResolutionPL, RandomPatchPL, RealImagePL
from pretext import RandomPretextConverter
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.447, 0.440, 0.407], [0.260, 0.256, 0.271])
])

# load test data
data_train = 'data/supervised/train'
data_test = 'data/supervised/test'

trainset = torchvision.datasets.ImageFolder(data_train, transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True)

testset = torchvision.datasets.ImageFolder(data_test, transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

model = editor18(3)
model_path = 'model/CENC/STL-10/exp2/E_epoch_199'
model.load_state_dict(torch.load(model_path))


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


net = torch.nn.Sequential(
    *list(model.children())[:1],  # only encoder
    Flatten(),
    torch.nn.Linear(6*6*512, 10)
)
for child in net.children():
    for param in child.parameters():
        param.requires_grad = False
    break
net = net.to(device)
param_to_update = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.Adam(param_to_update, lr=0.001)

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, 10)
model_conv = model_conv.to(device)
optimizer_conv = torch.optim.Adam(model_conv.fc.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()


def train(model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss


def test(model):
    model.eval()
    correct = 0.0
    total = 0.0
    for data in testloader:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct/total


for epoch in range(100):
    our_loss = train(net, optimizer, criterion)
    res_loss = train(model_conv, optimizer_conv, criterion)
    our_acc = test(net)
    res_acc = test(model_conv)
    print('Epoch %d: Tansen [loss: %.3f, acc: %2d %%]; Resnet18 [loss: %.3f, acc: %2d %%]' % (
        epoch+1, our_loss, 100*our_acc, res_loss, 100*res_acc))
