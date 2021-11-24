# Image-Search-Using-Convolutional-Neural-Network

This project is part of my work during research intern at Indian Institute of Technology (IIT - Bombay)(May 2017 - July 2017). 

Model is trained on the CIFAR dataset using Convolutional Neural Network to classify images and search on the web for corresponding image. It can be easily extended to make the search engine more inclusive with more data and image type.

## Technicalities

* Data Load and Transformation: 
  * Training data was loaded from the CIFAR dataset and transformed to be normalized with a batch size of 4 and shuffling enabled.
* Defining the Convolutional Neural Network :
  The Network was defined to carry out seven convolution with batch normalization after each convolution followed by three linear fully connected layers.
  During the forward propagation, rectified linear unit(ReLu) was and max pooling were used to extract the features from the image.
              
              
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
                   
                    x = x.view(-1, 1*1*256)
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
                    
* Define a Loss function and optimizer
* Train the network
* Test the network on the test data
 







