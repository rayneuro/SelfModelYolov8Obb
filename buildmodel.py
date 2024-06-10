import torch
import torch.nn as nn

from conv import *
from  block import *
from head import *




# Build Yolov8 obb model 

class ModelYolov8obb(nn.Module):

    def __init__(self, nc=80, ch = 3, arch=None, act=None): # number of classed , input channels
        """
        YOLOv8 model.

        Args:
            nc (int): Number of classes
            anchors (list): Anchors
            ch (list): Channels
            arch (str): Architecture
            act (str): Activation
        """
        super(ModelYolov8obb, self).__init__()


        
        self.names = {i: f"{i}" for i in range(self.nc)}  # default names dict
        # backbone
        self.conv1 = Conv(3, 64, 3, 2) # l0  80x320x320
        self.conv2 = Conv(64, 128, 3, 2) #l1  123x160x160x
        self.c2f1 =  C2f(128, 128, n=3, shortcut=True, g=1, e=0.5) #l2  
        self.conv3 = Conv(128, 256, 1, 1) #l3 
        self.c2f2 = C2f(256,256, n = 6,shortcut = True,g =1, e = 0.5) # l4 Concat stride = 8
        self.conv4 = C2f(256,512, ) # l5
        self.c2f3 = C2f(512,512, n = 6, shortcut = True, g = 1, e = 0.5) #l6 Concat stride = 16
        self.conv5 = Conv(512, 1024, 3, 2) # l7
        self.c2f4 = C2f(1024, 1024, n = 3, shortcut = True, g = 1, e  = 0.5)  # l8
        self.sppf = SPPF(1024, 1024,k=5) # l9 Concat stride = 32

        # head 
        self.Upsample1 = nn.Upsample(size =None,scale_factor=2, mode='nearest') # l10
        self.concat1 = Concat(dimension = 1) # l11
        self.c2f5 = C2f(512,512, n = 3, shortcut = False, g = 1, e = 0.5) # l12
        self.Upsample2 = nn.Upsample(size =None,scale_factor=2, mode='nearest') # l13
        self.concat2 = Concat(dimension = 1) # l14
        self.c2f6 =  C2f(256,256, n = 3, shortcut = False, g = 1, e = 0.5) # l15
        self.conv6 = Conv(256, 256, 3, 2) # l16
        self.concat3 = Concat(dimension = 1)   # l17
        self.c2f7 = C2f(512,512, n = 3, shortcut = False, g = 1, e = 0.5) # l18
        self.conv7 = Conv(512, 512, 3, 2) # l19
        self.concat4 = Concat(dimension = 1) # l20
        self.c2f8 = C2f(1024, 1024, n = 3, shortcut = False, g = 1, e = 0.5) # l21
        self.detecthead  = OBB(nc = nc, ne = 1, ch = (256,512,512)) # l22 

    
    def forward(self, x):
        x = self.conv1(x) # l0
        x = self.conv2(x) # l1
        x = self.c2f1(x) # l2   
        x = self.conv3(x) # l3
        l4x = self.c2f2(x) # l4
        x = self.conv4(l4x)   # l5
        l6x = self.c2f3(x)   # l6
        x = self.conv5(l6x)   # l7
        x = self.c2f4(x)    # l8
        l9x = self.sppf(x)    # l9
        x = self.Upsample1(l9x)   # l10
        x = self.concat1([x,l6x]) # l11
        l12x = self.c2f5(x) # l12
        x = self.Upsample2(l12x) # l13
        x = self.concat1([x,l4x]) # l14
        l15x = self.c2f6(x) # l15
        x = self.conv6(l15x) # l16
        x = self.concat3([x,l12x]) # l17
        l18x = self.c2f7(x) # l18
        x = self.conv7(l18x) # l19 
        x = self.concat4([x,l9x]) # l20 
        l21x = self.c2f8(x) # l21
        
        return self.detecthead([l21x,l18x,l15x])