from torchvision import models
import torch.nn as nn

def discriminator(model_name='resnet18',pretrained=True, **kwargs):
  # get pretrained model
  if model_name == 'resnet18':
      model_pre = models.resnet18(pretrained)
  elif model_name == 'resnet34':
      model_pre = models.resnet34(pretrained)
  elif model_nam e == 'resnet50':
      model_pre = models.resnet50(pretrained)
  elif model_name == 'resnet101':
      model_pre = models.resnet101(pretrained)
  elif model_name == 'resnet152':
      model_pre = models.resnet152(pretrained)
  else:
      raise ValueError('Unknown model type')
  # Freeze trained model weights for parameter transfer learning
  for param in model_pre.parameters():
    param.requires_grad = False
  # Add custom classifier:
  num_features = model_pre.fc.in_features
# model.fc = SigmoidLinear(num_features, num_classes)
  model_pre.fc = nn.Sequential(
                        nn.Linear(num_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(256, 2),
                        nn.LogSoftmax(dim=1))
  return model_pre