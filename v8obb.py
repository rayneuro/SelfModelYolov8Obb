import buildmodel
import loaddata
import numpy as np
import torch
import loss



v8obb = buildmodel.ModelYolov8obb(nc = 80)

for module in v8obb.modules():
    print(module)

# optimizer = torch.optim.Adam(v8obb.parameters(), lr=lr)








'''
epoch = 2
total_samples = len(dataset)
n_iterations = np.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(epoch):
'''

#test Input Tensor




