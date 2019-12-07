from torchsummary import summary
from model import editor18

model = editor18(3)
summary(model.cuda(), (3, 96, 96))
