import torch
import torchvision
from torchvision import transforms
from editor import editor18
import matplotlib.pyplot as plt

model = editor18(3)
model_path = 'model/CENC/STL-10/exp1/E_epoch_99'
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.447, 0.440, 0.407], [0.260, 0.256, 0.271])
])

# load test data
data_dir = 'data/supervised/test'
test_dataset = torchvision.datasets.ImageFolder(data_dir, transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# testing some sample
def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(*tensor.size())
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(5) 

dataiter = iter(testloader)
images, labels = dataiter.next()

for i,img in enumerate(images):
    imshow(model(img), str(labels[i]))
