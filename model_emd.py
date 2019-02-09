import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision
import math
from forest import Forest
#from ops import AngleLinear1 as AngleLinear
from ops import AngleLinear, CenterExclusiveAngleLinear, NormedLinear, CenterlossExclusiveLinear, ExclusiveLinear
#==================================================================#
# the base architecture
#==================================================================#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class VGG16(nn.Module):

    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes, bias=False)

        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):

        x = self.vgg(x)
        x = self.classifier(x)

        return torch.softmax(x, dim=1)        
    
class VGG16_dt(nn.Module):

    def __init__(self, num_classes, num_trees=10, tree_depth=10):
        super(VGG16_dt, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=True)
        self.classifier = Forest(1000, num_trees, tree_depth, num_classes)

    def forward(self, x):

        x = self.vgg(x)
        x = self.classifier(x)

        return x   
        
class Resnet20(nn.Module):
    def __init__(self, dim=512, bn=True, num_age=None):
        super(Resnet20, self).__init__()
        self.dim = dim
        self.bn = bn
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*8*8, dim)

        # added by xu gang to make the layer solve two kinds of problems
        self.age_classification = nn.Linear(dim, num_age)
        
        self.Softmax = nn.Softmax(dim=1)

        # KEEP_PROB = 1.0
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(KEEP_PROB)
        # self.fc6 = nn.Linear(512*15*15, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.new_1 = nn.Linear(4096, 256)
        # self.new_2 = nn.Linear(256, num_identity)
        # self.age_fc6 = nn.Linear(512*15*15, 4096)
        # self.age_fc7 = nn.Linear(4096, 4096)
        # self.age_fc8 = nn.Linear(4096, num_age)

        if self.bn:
          self.bn1_1 = nn.BatchNorm2d(64)
          self.bn1_2 = nn.BatchNorm2d(64)
          self.bn1_3 = nn.BatchNorm2d(64)
          self.bn2_1 = nn.BatchNorm2d(128)
          self.bn2_2 = nn.BatchNorm2d(128)
          self.bn2_3 = nn.BatchNorm2d(128)
          self.bn2_4 = nn.BatchNorm2d(128)
          self.bn2_5 = nn.BatchNorm2d(128)
          self.bn3_1 = nn.BatchNorm2d(256)
          self.bn3_2 = nn.BatchNorm2d(256)
          self.bn3_3 = nn.BatchNorm2d(256)
          self.bn3_4 = nn.BatchNorm2d(256)
          self.bn3_5 = nn.BatchNorm2d(256)
          self.bn3_6 = nn.BatchNorm2d(256)
          self.bn3_7 = nn.BatchNorm2d(256)
          self.bn3_8 = nn.BatchNorm2d(256)
          self.bn3_9 = nn.BatchNorm2d(256)
          self.bn4_1 = nn.BatchNorm2d(512)
          self.bn4_2 = nn.BatchNorm2d(512)
          self.bn4_3 = nn.BatchNorm2d(512)

    def forward(self, x):
        if self.bn:
            x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
            x = x + self.relu1_3(self.bn1_3(self.conv1_3(self.relu1_2(self.bn1_2(self.conv1_2(x))))))
            stage1 = x

            x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
            x = x + self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x))))))
            x = x + self.relu2_5(self.bn2_5(self.conv2_5(self.relu2_4(self.bn2_4(self.conv2_4(x))))))
            stage2 = x

            x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
            x = x + self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x))))))
            x = x + self.relu3_5(self.bn3_5(self.conv3_5(self.relu3_4(self.bn3_4(self.conv3_4(x))))))
            x = x + self.relu3_7(self.bn3_7(self.conv3_7(self.relu3_6(self.bn3_6(self.conv3_6(x))))))
            x = x + self.relu3_9(self.bn3_9(self.conv3_9(self.relu3_8(self.bn3_8(self.conv3_8(x))))))
            stage3 = x

            x = self.relu4_1(self.bn4_1(self.conv4_1(stage3)))
            x = x + self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x))))))
            stage4 = x


        else:
            x = self.relu1_1(self.conv1_1(x))
            x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
            stage1 = x

            x = self.relu2_1(self.conv2_1(x))
            x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
            x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
            stage2 = x

            x = self.relu3_1(self.conv3_1(x))
            x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
            x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
            x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
            x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
            stage3 = x

            x = self.relu4_1(self.conv4_1(stage3))
            x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
            stage4 = x

        flattened = stage4.view(-1, 512*8*8)

        fc5 = self.fc5(flattened)

        age_logits = self.age_classification(fc5)
        
        age_prob = self.Softmax(age_logits)

        return age_prob

class Resnet64(nn.Module):
    def __init__(self, dim=512):
        super(Resnet64, self).__init__()
        self.dim = dim
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)
        self.conv1_4 = nn.Conv2d(64,64,3,1,1)
        self.relu1_4 = nn.PReLU(64)
        self.conv1_5 = nn.Conv2d(64,64,3,1,1)
        self.relu1_5 = nn.PReLU(64)
        self.conv1_6 = nn.Conv2d(64,64,3,1,1)
        self.relu1_6 = nn.PReLU(64)
        self.conv1_7 = nn.Conv2d(64,64,3,1,1)
        self.relu1_7 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)

        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)
        self.conv2_4 = nn.Conv2d(128,128,3,1,1)
        self.relu2_4 = nn.PReLU(128)

        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)
        self.conv2_6 = nn.Conv2d(128,128,3,1,1)
        self.relu2_6 = nn.PReLU(128)

        self.conv2_7 = nn.Conv2d(128,128,3,1,1)
        self.relu2_7 = nn.PReLU(128)
        self.conv2_8 = nn.Conv2d(128,128,3,1,1)
        self.relu2_8 = nn.PReLU(128)

        self.conv2_9 = nn.Conv2d(128,128,3,1,1)
        self.relu2_9 = nn.PReLU(128)
        self.conv2_10 = nn.Conv2d(128,128,3,1,1)
        self.relu2_10 = nn.PReLU(128)

        self.conv2_11 = nn.Conv2d(128,128,3,1,1)
        self.relu2_11 = nn.PReLU(128)
        self.conv2_12 = nn.Conv2d(128,128,3,1,1)
        self.relu2_12 = nn.PReLU(128)

        self.conv2_13 = nn.Conv2d(128,128,3,1,1)
        self.relu2_13 = nn.PReLU(128)
        self.conv2_14 = nn.Conv2d(128,128,3,1,1)
        self.relu2_14 = nn.PReLU(128)
        self.conv2_15 = nn.Conv2d(128,128,3,1,1)
        self.relu2_15 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv3_10 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_10 = nn.PReLU(256)
        self.conv3_11 = nn.Conv2d(256,256,3,1,1)
        self.relu3_11 = nn.PReLU(256)

        self.conv3_12 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_12 = nn.PReLU(256)
        self.conv3_13 = nn.Conv2d(256,256,3,1,1)
        self.relu3_13 = nn.PReLU(256)

        self.conv3_14 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_14 = nn.PReLU(256)
        self.conv3_15 = nn.Conv2d(256,256,3,1,1)
        self.relu3_15 = nn.PReLU(256)

        self.conv3_16 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_16 = nn.PReLU(256)
        self.conv3_17 = nn.Conv2d(256,256,3,1,1)
        self.relu3_17 = nn.PReLU(256)

        self.conv3_18 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_18 = nn.PReLU(256)
        self.conv3_19 = nn.Conv2d(256,256,3,1,1)
        self.relu3_19 = nn.PReLU(256)

        self.conv3_20 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_20 = nn.PReLU(256)
        self.conv3_21 = nn.Conv2d(256,256,3,1,1)
        self.relu3_21 = nn.PReLU(256)

        self.conv3_22 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_22 = nn.PReLU(256)
        self.conv3_23 = nn.Conv2d(256,256,3,1,1)
        self.relu3_23 = nn.PReLU(256)

        self.conv3_24 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_24 = nn.PReLU(256)
        self.conv3_25 = nn.Conv2d(256,256,3,1,1)
        self.relu3_25 = nn.PReLU(256)

        self.conv3_26 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_26 = nn.PReLU(256)
        self.conv3_27 = nn.Conv2d(256,256,3,1,1)
        self.relu3_27 = nn.PReLU(256)

        self.conv3_28 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_28 = nn.PReLU(256)
        self.conv3_29 = nn.Conv2d(256,256,3,1,1)
        self.relu3_29 = nn.PReLU(256)

        self.conv3_30 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_30 = nn.PReLU(256)
        self.conv3_31 = nn.Conv2d(256,256,3,1,1)
        self.relu3_31 = nn.PReLU(256)

        self.conv3_32 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_32 = nn.PReLU(256)
        self.conv3_33 = nn.Conv2d(256,256,3,1,1)
        self.relu3_33 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)
        self.conv4_4 = nn.Conv2d(512,512,3,1,1) #=>B*512*7*6
        self.relu4_4 = nn.PReLU(512)
        self.conv4_5 = nn.Conv2d(512,512,3,1,1)
        self.relu4_5 = nn.PReLU(512)
        self.conv4_6 = nn.Conv2d(512,512,3,1,1)
        self.relu4_6 = nn.PReLU(512)
        self.conv4_7 = nn.Conv2d(512,512,3,1,1)
        self.relu4_7 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6, self.dim)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = x + self.relu1_5(self.conv1_5(self.relu1_4(self.conv1_4(x))))
        x = x + self.relu1_7(self.conv1_7(self.relu1_6(self.conv1_6(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = x + self.relu2_7(self.conv2_7(self.relu2_6(self.conv2_6(x))))
        x = x + self.relu2_9(self.conv2_9(self.relu2_8(self.conv2_8(x))))
        x = x + self.relu2_11(self.conv2_11(self.relu2_10(self.conv2_10(x))))
        x = x + self.relu2_13(self.conv2_13(self.relu2_12(self.conv2_12(x))))
        x = x + self.relu2_15(self.conv2_15(self.relu2_14(self.conv2_14(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = x + self.relu3_11(self.conv3_11(self.relu3_10(self.conv3_10(x))))
        x = x + self.relu3_13(self.conv3_13(self.relu3_12(self.conv3_12(x))))
        x = x + self.relu3_15(self.conv3_15(self.relu3_14(self.conv3_14(x))))
        x = x + self.relu3_17(self.conv3_17(self.relu3_16(self.conv3_16(x))))
        x = x + self.relu3_19(self.conv3_19(self.relu3_18(self.conv3_18(x))))
        x = x + self.relu3_21(self.conv3_21(self.relu3_20(self.conv3_20(x))))
        x = x + self.relu3_23(self.conv3_23(self.relu3_22(self.conv3_22(x))))
        x = x + self.relu3_25(self.conv3_25(self.relu3_24(self.conv3_24(x))))
        x = x + self.relu3_27(self.conv3_27(self.relu3_26(self.conv3_26(x))))
        x = x + self.relu3_29(self.conv3_29(self.relu3_28(self.conv3_28(x))))
        x = x + self.relu3_31(self.conv3_31(self.relu3_30(self.conv3_30(x))))
        x = x + self.relu3_33(self.conv3_33(self.relu3_32(self.conv3_32(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = x + self.relu4_5(self.conv4_5(self.relu4_4(self.conv4_4(x))))
        x = x + self.relu4_7(self.conv4_7(self.relu4_6(self.conv4_6(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        return x

#==================================================================#
# baseline model with cross-entropy loss
#==================================================================#
class Baseline(nn.Module):
    def __init__(self, dim=512, num_age=None):
        super(Baseline, self).__init__()
        self.base = Resnet20(num_age=num_age)
        self.base.apply(weights_init)

    def forward(self, x):
        age_logits = self.base(x)
        return age_logits

class VGG16_Baseline(nn.Module):
    def __init__(self, num_age=None):
        super(VGG16_Baseline, self).__init__()
        self.base = VGG16(num_age)

    def forward(self, x):
        age_logits = self.base(x)
        return age_logits
    
class VGG16_dt_Baseline(nn.Module):
    def __init__(self, num_age=None):
        super(VGG16_dt_Baseline, self).__init__()
        self.base = VGG16_dt(num_age)

    def forward(self, x):
        age_logits = self.base(x)
        return age_logits