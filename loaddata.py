import torch
import torch.nn as nn
import cv2
import os
from tqdm import tqdm
from torch.utils.data import dataloader , distributed , Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from typing import List
from augment import * 





def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )





class Yolov8obbDataset(Dataset) : 
    def __init__(self, image_folder, label_folder, nc = 3, transform = None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.images = load_images_from_folder(self.image_folder)
        self.labels = get_label_from_folder(self.label_folder)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        #load images and bounding boxes label

        label = deepcopy(self.labels[idx])
        cls = deepcopy(label[0])
        image = self.images[idx]
        
        #if self.transform:
        #image = self.transform(image)

        target = {}
        target['boxes'] = torch.tensor([labels[1:7]],dtype = torch.float32)
        target['labels'] = torch.tensor([cls],dtype = torch.int64)
        target['h'] =  torch.tensor(image.shape[2],dtype = torch.int64)
        target['w'] =  torch.tensor(image.shape[1],dtype = torch.int64)
        
        return image, target
    
    
    def collate_fn(batch):
        return tuple(zip(*batch))





# Get cls, x1,y1,x2,y2,x3,y3,x4,y4  from the label file
def get_label_from_folder(folder) -> List[List[List[float]]]:
    
    image_labels = []
    for file in sorted(os.listdir(folder)):
        with open(folder + '/' + file,'r') as f:
            lines = f.read().splitlines()
            labels = []
            print('Loading labels: ... ')
            for line in tqdm(lines):
                label = line.split(' ')
                label = [float(x) for x in label] # x1,y1,x2,y2,x3,y3,x4,y4
                label[0] = int(label[0]) # cls
                labels.append(label)
        image_labels.append(labels)
    return image_labels


def LabelsToTensor(labels):
    pass


    
def load_images_from_folder(folder) -> torch.Tensor:
    images = []
    print('Loading images: ... ')
    trans = transforms.ToTensor()
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = trans(img)
        if img is not None:
            images.append(img) 
    # images == List[Torch.Tensor]
    return images_to_tensor(images)


def _to_batch(batch_size, _img_size , imgs: List[torch.Tensor], device) -> torch.Tensor: # Get the bacth tensor from the list of images
    batch_list = [x.unsqueeze(0) for x in imgs]
    if batch_size > len(batch_list):
        fill_size = batch_size - len(batch_list)
        batch_list.append(torch.zeros([fill_size, 3, _img_size[0], _img_size[1]]).to(device))
    batch = torch.cat(batch_list, 0).half()
    return batch

def images_to_tensor(images) -> torch.Tensor:
    image_list = [x.unsqueeze(0) for x in images]
    
    images_tensor = torch.cat(image_list, 0).half()
    return images_tensor
    


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


labels = get_label_from_folder('./Labels')
print(type(labels))
print(labels[0][0][1:9])


#TrainDataLoader = dataloader(TrainDataset,batch_size = 1,shuffle = True,num_workers = 4)


#TrainDataloader = Yolov8obbDataLoader(TrainDataset,batch_size = 32,shuffle = True,num_workers = 4)











