import cv2
import torch
import torchvision.transforms as transforms




class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    

model = Model()

print(model.model[-1])

