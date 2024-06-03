import cv2
import torch
import torchvision.transforms as transforms


img = cv2.imread('./Images/32TL600_W1_GC1_R1_P14_G35_4.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

trans = transforms.ToTensor()  # Define a transformation to convert cv2 image to tensor

img = trans(img) # convert cv2 image to tensor

print(img)
print(img.shape)