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
from dataset import *



def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]






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
        
    


















