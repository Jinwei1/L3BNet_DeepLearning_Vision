from torchvision import models
import torch.nn as nn
import pdb
import torch

        
def resnet18(pretrained = True):
    return models.resnet18(pretrained)

def vgg16(pretrained = True):
    return models.vgg16(pretrained)

def alexnet(pretrained = True):
    return models.alexnet(pretrained)


class My_Model(nn.Module):
    def __init__(self, nClasses = 200):
        super(My_Model,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,stride=1, padding = 1)
        self.relu_T = nn.ReLU(True);
        self.batch_norm_32 = nn.BatchNorm2d(32);
        self.pool_2_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_32_64_5 = nn.Conv2d(32,64,kernel_size=5,stride=1, padding = 2)
        self.conv_32_64_3 = nn.Conv2d(32,64,kernel_size=3,stride=1, padding = 1)
        # self.relu_1_2 = nn.ReLU(True);
        self.batch_norm_64 = nn.BatchNorm2d(64);
        # self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2);

        # self.conv_32_64_7 = nn.Conv2d(32,64,kernel_size=7,stride=1, padding = 3);
        # self.relu_1_3 = nn.ReLU(True);
        # self.batch_norm_3 = nn.BatchNorm2d(64);

        self.conv_32_32_1 = nn.Conv2d(32,32,kernel_size=1,stride=1, padding = 0);
        self.conv_32_64_1 = nn.Conv2d(32,64,kernel_size=1,stride=1, padding = 0);
        # self.relu_1_3 = nn.ReLU(True);
        # self.batch_norm_3 = nn.BatchNorm2d(64);
        self.conv_64_64_1 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0);
        self.conv_64_64_3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1);

        self.conv_64_128_1 = nn.Conv2d(64,128,kernel_size=1,stride=1, padding = 0)
        self.conv_64_128_3 = nn.Conv2d(64,128,kernel_size=3,stride=1, padding = 1)
        self.conv_128_128_1 = nn.Conv2d(128,128,kernel_size=1,stride=1, padding = 0)
        self.conv_128_128_3 = nn.Conv2d(128,128,kernel_size=3,stride=1, padding = 1)
        self.conv_128_256_3 = nn.Conv2d(128,256,kernel_size=3,stride=1, padding = 1)
        # self.relu_1_2 = nn.ReLU(True);
        self.batch_norm_128 = nn.BatchNorm2d(128);
        self.batch_norm_256 = nn.BatchNorm2d(256);

        self.fc_1 = nn.Linear(16*16*256, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm1d(1024);
        self.dropout = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        #y = self.conv_2(x)
        
        y = self.conv_1(x)
        y = self.relu_T(y)
        y = self.batch_norm_32(y)
        y = self.pool_2_2(y)
        y = self.conv_32_32_1(y)

       
        y1_1 = self.conv_32_64_3(y)
        y1_1 = self.relu_T(y1_1)
        # y1_1 = self.batch_norm_64(y1_1)
        y1_1 = self.dropout(y1_1)
        

        
        y1_3 = self.conv_32_64_1(y)
        y1_3 = self.relu_T(y1_3)
        # y1_1 = self.batch_norm_64(y1_1)
        y1_3 = self.dropout(y1_3)

        y = y1_1 + y1_3
        y = self.batch_norm_64(y)
        y = self.dropout(y)
        y = self.conv_64_64_1(y)

        
        y2_1 = self.conv_64_128_3(y)
        y2_1 = self.relu_T(y2_1)
        # y1_1 = self.batch_norm_64(y1_1)
        y2_1 = self.dropout(y2_1)
        

       
        y2_3 = self.conv_64_128_1(y)
        y2_3 = self.relu_T(y2_3)
        # y1_1 = self.batch_norm_64(y1_1)
        y2_3 = self.dropout(y2_3)

        y = y2_1 + y2_3
        y = self.batch_norm_128(y)
        y = self.dropout(y)
        y = self.conv_128_128_1(y)

        y = self.conv_128_256_3(y) 
        # y = self.conv_128_128_3(y) 
        y = self.relu_T(y)
        y = self.batch_norm_256(y)
        y = self.pool_2_2(y)
       

        # y = self.batch_norm_3(y)
        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.dropout(y)
        y = self.fc_2(y)
        return(y)


def my_model():
    return My_Model();

class Line_Model(nn.Module):
    def __init__(self, nClasses = 200):
        super(Line_Model,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,stride=1, padding = 1)
        self.relu_T = nn.ReLU(True);
        self.batch_norm_32 = nn.BatchNorm2d(32);
        self.pool_2_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_32_64_3 = nn.Conv2d(32,64,kernel_size=3,stride=1, padding = 1)
        # self.relu_1_2 = nn.ReLU(True);
        self.batch_norm_64 = nn.BatchNorm2d(64);
        # self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2);

        self.conv_64_128_3 = nn.Conv2d(64,128,kernel_size=3,stride=1, padding = 1)
        self.conv_128_128_3 = nn.Conv2d(128,128,kernel_size=3,stride=1, padding = 1)
        self.conv_128_256_3 = nn.Conv2d(128,256,kernel_size=3,stride=1, padding = 1)
        self.conv_256_256_3 = nn.Conv2d(256,256,kernel_size=3,stride=1, padding = 1)
        # self.relu_1_2 = nn.ReLU(True);
        self.batch_norm_128 = nn.BatchNorm2d(128);
        self.batch_norm_256 = nn.BatchNorm2d(256);

        # self.conv_32_64_7 = nn.Conv2d(32,64,kernel_size=7,stride=1, padding = 3);
        # self.relu_1_3 = nn.ReLU(True);
        # self.batch_norm_3 = nn.BatchNorm2d(64);

        self.conv_32_32_1 = nn.Conv2d(32,32,kernel_size=1,stride=1, padding = 0);
        self.conv_32_64_1 = nn.Conv2d(32,64,kernel_size=1,stride=1, padding = 0);
        # self.relu_1_3 = nn.ReLU(True);
        # self.batch_norm_3 = nn.BatchNorm2d(64);
        self.conv_64_64_1 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0);
        self.conv_64_64_3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1);

        self.fc_1 = nn.Linear(4*4*256, 2048);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.dropout = nn.Dropout(p = 0.5);

        self.fc_1_2 = nn.Linear(2048,2048); #new
        self.relu_4 = nn.ReLU(True);#new

        self.softmax = nn.Softmax(dim=0);#new
        self.dropout = nn.Dropout(p = 0.5);#new
        
        self.batch_norm_4 = nn.BatchNorm1d(2048);
        self.fc_2 = nn.Linear(2048, nClasses);
        
        
    def forward(self,x):
        #pdb.set_trace();
        #y = self.conv_2(x)
        
        y = self.conv_1(x)
        y = self.relu_T(y)
        y = self.batch_norm_32(y)
        y = self.pool_2_2(y)

       
        y = self.conv_32_64_3(y)
        y = self.conv_64_64_3(y)
        y = self.relu_T(y)
        y = self.batch_norm_64(y)
        y = self.pool_2_2(y)

        y = self.conv_64_128_3(y) 
        y = self.conv_128_128_3(y) 
        y = self.relu_T(y)
        y = self.batch_norm_128(y)
        y = self.pool_2_2(y)

        # add new
        y = self.conv_128_256_3(y) 
        y = self.conv_256_256_3(y) 
        y = self.relu_T(y)
        y = self.batch_norm_256(y)
        y = self.pool_2_2(y)

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
 
        y = self.batch_norm_4(y)
        y = self.dropout(y)
        y = self.fc_2(y)
        
        return(y)

def line_model():
    return Line_Model();


