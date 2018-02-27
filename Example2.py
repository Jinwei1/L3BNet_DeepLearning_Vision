#Demo for CS7-GV1

#general modules
# some codes followed the tutorial on https://github.com/pytorch/examples/blob/master/imagenet/main.py 
from __future__ import print_function, division
import os
import argparse
import time
import copy
import sys
import numpy as np

from graphviz import Digraph
import re
import torchvision.models as models

#pytorch modules
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.autograd import Variable
import pdb

#user defined modules
import Augmentation as ag
import Models
from TestC import Test
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='CS7-GV1 Final Project');


#add/remove arguments as required. It is useful when tuning hyperparameters from bash scripts
parser.add_argument('--aug', type=str, default = '', help='data augmentation strategy')
# parser.add_argument('--optim', type=str, default = '', help='optim strategy')
parser.add_argument('--datapath', type=str, default='', 
               help='root folder for data.It contains two sub-directories train and val')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')               
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--batch_size', type=int, default = 128,
                    help='batch size')
parser.add_argument('--model', type=str, default = None, help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--tag', type=str, default=None,
                    help='unique_identifier used to save results')
args = parser.parse_args();
if not args.tag:
    print('Please specify tag...')
    exit()

orig_stdout = sys.stdout
# f = open(args.tag+'.logs', 'a')
f = open(args.tag+'.test.logs', 'a')
# sys.stdout = f
print (args)

#Define augmentation strategy
augmentation_strategy = ag.Augmentation(args.aug);
data_transforms = augmentation_strategy.applyTransforms();
##

#Root directory
data_dir = args.datapath;
##

######### Data Loader ###########
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=16) # set num_workers higher for more cores and faster data loading
             for x in ['train', 'val']}
                 
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
#################################
print(dsets['train'].classes)
#set GPU flag
use_gpu = args.cuda;
##
#Load model . Once you define your own model in Models.py, you can call it from here. 
if args.model == "ResNet18":
    current_model = Models.resnet18(args.pretrained)
    num_ftrs = current_model.fc.in_features
    current_model.fc = nn.Linear(num_ftrs, len(dset_classes));
elif args.model == "vgg16":
    current_model = Models.vgg16(args.pretrained)
    
elif args.model == "alexnet":
    current_model = Models.alexnet(args.pretrained)
    
elif args.model == 'Demo':
    current_model = Models.demo_model();

elif args.model == 'MyModel':
    if args.pretrained:
        current_model = torch.load(args.tag+'.model');
    else:
        current_model = Models.my_model();

elif args.model == 'LineModel':
    if args.pretrained:
        current_model = torch.load(args.tag+'.model');
    else:
        current_model = Models.line_model();
    print(current_model)

else :
    print ("Model %s not found"%(args.model))
    exit();    


if use_gpu:
	current_model = current_model.cuda();
	print('using GPU....');


    
# uses a cross entropy loss as the loss function
# http://pytorch.org/docs/master/nn.html#
# use a log loss
criterion = nn.CrossEntropyLoss()
# criterion = nn.PoissonNLLLoss()

#uses stochastic gradient descent for learning
# http://pytorch.org/docs/master/optim.html
if args.model == 'alexnet':
     # adam optimizer doesn't work for alexnet here, 0.005 acc, random guess
    optimizer_ft = optim.SGD(current_model.parameters(), lr=args.lr, momentum=0.9)
elif args.model =='ResNet18':
    optimizer_ft = optim.SGD(current_model.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer_ft = optim.Adam(current_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#the learning rate condition. The ReduceLROnPlateau class reduces the learning rate by 'factor' after 'patience' epochs.
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.2,patience = 4, verbose = True)

def test_model(model):
    model.train(False)
    # Iterate over data.
    for count, data in enumerate(dset_loaders[phase]):
        # get the inputs
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        arr = preds.data.cpu().numpy()
        np.savetxt(args.tag+'_output.csv', arr)

# def make_dot(var):
#     node_attr = dict(style='fillSmoothL1Loss()ed',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()

#     def add_nodes(var):
#         if var not in seen:
#             if isinstance(var, Variable):
#                 value = '('+(', ').join(['%d'% v for v in var.size()])+')'
#                 dot.node(str(id(var)), str(value), fillcolor='lightblue')
#             else:
#                 dot.node(str(iprec1, prec5 = accuracy
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()

#     def add_nodes(var):
#         if var not in seen:
#             if isinstance(var, Variable):
#                 value = '('+(', ').join(['%d'% v for v in(output.data, target, topk=(1, 5))d(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'previous_functions'):
#                 for u in var.previous_functions:
#                     dot.edge(str(id(u[0])), str(id(var)))
#                     add_nodes(u[0])
#     add_nodes(var.grad_fn)
#     return dot  
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

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_top1 = TopAcc()
            running_top5 = TopAcc()
         
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data
                labels0 = labels.cuda(async=True)
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
	
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data,1)

                loss =  criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                pred1, pred5 = accuracy(outputs.data,labels0,topk=(1,5))
                # print(outputs)
                # g = make_dot(outputs)

                # statistics
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                running_top1.update( pred1[0],inputs.size(0))
                running_top5.update( pred5[0],inputs.size(0))
                #if count%10 == 0:
                #    print('Batch %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1,running_loss/(args.batch_size*(count+1)),running_corrects/(args.batch_size*(count+1))))
                #print('Running Loss = %0. model(x6f'%(running_loss/(args.batch_size*(count+1))))
            # print('%d / %d',running_corrects,dset_sizes[phase])
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            epoch_top1 = running_top1.avg / 100
            epoch_top5 = running_top5.avg /100

            print('Epoch %d || %s Loss: %.4f || Acc,Top1: %.4f || Top5: %.4f '%(epoch,
                phase, epoch_loss, epoch_acc, epoch_top5),end = ' || ')
         
            #pdb.set_trace();
            if phase == 'val':
                print ('\n', end='');
                # f.write('\n');
                lr_scheduler.step(epoch_loss);
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
   
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model


#comment the block below if you are not training 
######################
trained_model = train_model(current_model, criterion, optimizer_ft, scheduler_ft,
                      num_epochs=args.epochs);
with open(args.tag+'.model', 'wb') as f:
    torch.save(trained_model, f);
# with open(args.tag+'.parameter', 'wb') as f:
#     torch.save(trained_model.load_state_dict, f);
######################    
## uncomment the lines blow while testing and generating confusion matrix 
##  for 20 random classess in validation dataset


# trained_model = current_model;
# testDataPath = args.datapath
# t = Test(args.aug,trained_model);
# scores = t.testfromdir(testDataPath);

######################  

# sys.stdout = orig_stdout
f.closed