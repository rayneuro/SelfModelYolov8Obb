import torch
import torch.nn as nn
import cv2
import os
from tqdm import tqdm
from torch.utils.data import dataloader , distributed , Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from typing import List
from augment import * 



class Yolov8obbDataset(Dataset) : 
    def __init__(self, image_folder, label_folder,transform = None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.images = load_images_from_folder(self.image_folder)
        self.labels = get_label_from_files(self.label_folder)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        #load images and bounding boxes label

        image = self.images[idx]
        labels = torch.tensor(self.labels[idx][0],dtype = torch.float32)
        if self.transform:
            image = self.transform(image)

        target = {}
        target['boxes'] = torch.tensor([labels[1:7]],dtype = torch.float32)
        target['labels'] = labels
        
        return image, target


# Get cls, x1,y1,x2,y2,x3,y3,x4,y4  from the label file
def get_label_from_files(file):
    with open(file,'r') as f:
        lines = f.read().splitlines()
        labels = []
        print('Loading labels: ... ')

        for line in tqdm(lines):
            label = line.split(' ')
            labels.append(label)
    return labels


def LabelsToTensor(labels):
    pass


    
def load_images_from_folder(folder):
    images = []
    print('Loading images: ... ')
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images


def _to_batch(batch_size, _img_size , imgs: List[torch.Tensor], device) -> torch.Tensor: # Get the bacth tensor from the list of images
    batch_list = [x.unsqueeze(0) for x in imgs]
    if batch_size > len(batch_list):
        fill_size = batch_size - len(batch_list)
        batch_list.append(torch.zeros([fill_size, 3, _img_size[0], _img_size[1]]).to(device))
    batch = torch.cat(batch_list, 0).half()
    return batch



def covert_to_batchtensor(images) -> torch.Tensor :
    trans = transforms.ToTensor()  # Define a transformation to convert cv2 image to tensor
    tensor_images = []
    for img in tqdm(images):
        tensor_images.append(trans(img)) # convert cv2 image to tensor

    batch = _to_batch(32,(640,640),tensor_images,torch.device('cuda'))
    return batch 


def rbox__xyconto_xywhr(rboxes):
    pass
        
    



class Yolov8obbDataLoader(dataloader.DataLoader):
    
    '''
        Dataloader for Yolov8obb worker

    '''
    def __init__(self,*arg,**kwargs):
        pass
        





# Yolov8 Data Augmentaion
def Yolov8Transforms():
    return transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor()
    ])



labels = get_label_from_files('./Labels/32TL600_W1_GC1_R1_P14_G35_4.txt')
print(type(labels))
print(labels)
#TrainDataLoader = dataloader(TrainDataset,batch_size = 1,shuffle = True,num_workers = 4)


#TrainDataloader = Yolov8obbDataLoader(TrainDataset,batch_size = 32,shuffle = True,num_workers = 4)











