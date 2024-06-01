import buildmodel
import loaddata

v8obb = buildmodel.ModelYolov8obb(nc = 80)

for module in v8obb.modules():
    print(module)



#test Input Tensor




