import os

#pytorch modules
import torch
from torchvision import datasets
from torch.autograd import Variable
import pdb
import torchvision.transforms as transforms
import Augmentation as ag
import numpy as np
from pandas_ml import ConfusionMatrix
# import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.metrics import confusion_matrix

class Test():
    def __init__(self, aug, model, use_gpu = True):
       #Define augmentation strategy
        self.augmentation_strategy = ag.Augmentation(aug);
        self.data_transforms = self.augmentation_strategy.applyTransforms();
        self.model = model;
        self.model.train(False)
        self.use_gpu = use_gpu

   

            
    def testfromdir(self,datapath,batch_size = 100):
       #Root directory
        data_dir = datapath;
        ##
        
        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                     for x in ['val']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=False, num_workers=16)# set num_workers higher for more cores and faster data loading
                     for x in ['val']}
        dsets_class2id = dsets['val'].class_to_idx
        scores = torch.cuda.FloatTensor();
        running_top1 = TopAcc()
        running_top5 = TopAcc()
        predList = []
        labelList = []
        predNameList = []
        labelNameList = []
        for count, data in enumerate(dset_loaders['val']):
            # get the inputs
            inputs, labels = data
            labels0 = labels.cuda(async=True)
            # wrap them in Variable
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)


            # forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            
            # scores = torch.cat((scores,torch.nn.functional.softmax(outputs).data),0);
            pred1, pred5 = accuracy(outputs.data,labels0,topk=(1,5))
            predList = np.append(predList,preds.cpu().numpy())
            labelList = np.append(labelList,labels.data.cpu().numpy())
            running_top1.update( pred1[0],inputs.size(0))
            running_top5.update( pred5[0],inputs.size(0))
        epoch_top1 = running_top1.avg / 100
        epoch_top5 = running_top5.avg / 100
        print(' \n Acc,Top1: %.4f || Top5: %.4f '%(epoch_top1, epoch_top5),end = ' || \n')
        # print(predList)
        # print(labelList)
        
        np.savetxt('outs/predList.csv',predList)
        np.savetxt('outs/labelList.csv',labelList)

        # print(dsets_class2id)
        import random
        confusion_matrix_0 = ConfusionMatrix(labelList, predList)
        ranLabesl = random.sample(range(0, 199), 20) 
      
        print('\n for these classess:')
        for name, idx in dsets_class2id.items():    # for name, age in list.items():  (for Python 3.x)
            aname = name
            for each in ranLabesl:
                if idx == each:
                    print(aname)
        cm = confusion_matrix(labelList, predList, labels = ranLabesl )
        print("Confusion matrix:\n%s" % cm)
      
        return 0;

class TopAcc(object):
    # compute and save topk accuracy
    def __init__(self):
        self.initial()
    def initial(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n):
        self.sum+=val
        self.count+=1
        self.avg = self.sum/self.count 

def accuracy(output, target, topk=(1,)):
    # calculate accuracy  using topk
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
