import os

#pytorch modules
import torch
from torchvision import datasets
from torch.autograd import Variable
import pdb
import torchvision.transforms as transforms
import Augmentation as ag
import numpy as np
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
        # dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
        #              for x in ['test']}
        dsets = datasets.ImageFolder(os.path.join(data_dir, 'test'), self.data_transforms['val'])

        dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=batch_size,
                                                       shuffle=False, num_workers=16)# set num_workers higher for more cores and faster data loading

        print(dsets.classes)
        scores = torch.cuda.FloatTensor();
        sum = []
        for count, data in enumerate(dset_loaders,0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if self.use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                # forward
                outputs = self.model(inputs)
                # scores = torch.cat((scores,torch.nn.functional.softmax(outputs).data),0);
                _, preds = torch.max(outputs.data, 1)
                arr =  preds.cpu().numpy()
                print(count)
                sum = np.append(sum,arr)
                print(arr)
                # print(sum)
                # f = open('results.csv', 'a')
                np.savetxt('results.csv', sum)
    
        return True;