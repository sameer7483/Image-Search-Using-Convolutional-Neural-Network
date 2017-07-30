import math
import random
import numpy as np
import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
#from sklearn.externals import joblib
import torch.utils.data as data_utils



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3,padding=1)
	self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)
	self.conv2_2 = nn.Conv2d(16,16,3,padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16,32,3,padding=1)
	self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,32,3,padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)

        self.conv5= nn.Conv2d(32,64,3,padding=1)
	self.conv5_2= nn.Conv2d(64,64,3,padding=1)
	self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6= nn.Conv2d(64,128,3,padding=1)
	self.conv6_bn = nn.BatchNorm2d(128)
	self.conv6_2= nn.Conv2d(128,128,3,padding=1)
	self.conv6_2_bn = nn.BatchNorm2d(128)
	self.conv7 = nn.Conv2d(128,256,3,padding=1)
	self.conv7_bn = nn.BatchNorm2d(256)

        
        
        self.fc1 = nn.Linear(1*1*256, 200)
        self.fc2 = nn.Linear(200, 75)
        self.fc3 = nn.Linear(75, 10)
	

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
	
        x = self.pool(F.relu(self.conv2_bn(self.conv2_2(self.conv2(x)))))
	
        x = (F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = (F.relu(self.conv5_bn(self.conv5_2(self.conv5(x)))))
        x = self.pool(F.relu(self.conv6(x)))
	x = F.relu(self.conv6_2_bn(self.conv6_2(x)))
	x = self.pool(F.relu(self.conv7_bn(self.conv7(x))))
	
	
        
        
        #x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 1*1*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Test:
	def go(self,image_path):
		net = Net()
		classes = ('plane', 'car', 'bird', 'cat',
           			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		net.load_state_dict(torch.load('nn1.pkl'))
		img = cv2.imread(image_path)
		img = cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA)
		img=img.transpose(2,0,1)
		tensor=torch.from_numpy(img)
		tensor=tensor.float()
		tensor=tensor.contiguous()
		images = Variable(tensor.view(-1,3,32,32))
		outputs = net(images)
		_, predicted = torch.max(outputs.data,1) 
		return classes[predicted[0]]


