# Image-Search-Using-Convolutional-Neural-Network

This project is part of my work during research intern at Indian Institute of Technology (IIT - Bombay)(May 2017 - July 2017). 

Model is trained on the CIFAR dataset using Convolutional Neural Network to classify images and search on the web for corresponding image. It can be easily extended to make the search engine more inclusive with more data and image type.

## Technicalities

### Data Load and Transformation: 
  * Training and Test data was loaded from the CIFAR dataset and transformed to be normalized with a batch size of 4 and shuffling enabled for training data and disabled for test data.
### Defining the Convolutional Neural Network :
  The Network was defined to carry out seven convolution with batch normalization after each convolution followed by three linear fully connected layers.
  During the forward propagation, Rectified linear unit(ReLu) and max pooling were used to extract the features from the image.
              
              
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
                    
### Define a Loss function and optimizer
  Cross Entropy Loss was used as the criterion with Adam as the optimizer and learning rate of 0.001.
  
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(net.parameters(), lr=0.001)
  
  
### Train the network
  The network was trained over an epoch cycle of 100, performing the backward propagation and updating the weights to get the optimized result on each output of forward propagation.
 

                     for epoch in range(100):  # loop over the dataset multiple times

                         running_loss = 0.0
                         for i, data in enumerate(trainloader, 0):
                             # get the inputs
                             inputs, labels = data

                             # wrap them in Variable
                             inputs, labels = Variable(inputs), Variable(labels)

                             # zero the parameter gradients
                             optimizer.zero_grad()

                             # forward + backward + optimize
                             outputs = net(inputs)
                             loss = criterion(outputs, labels)
                             loss.backward()
                             optimizer.step()

                             # print statistics
                             running_loss += loss.data[0]
                             if i % 2000 == 1999:    # print every 2000 mini-batches
                                 print('[%d, %5d] loss: %.3f' %
                                       (epoch + 1, i + 1, running_loss / 2000))
                                 running_loss = 0.0

                     print('Finished Training')


### Test the network on the test data:
  The model was tested on the test data and the final result of 92% accuracy was obtained using this model which was quite impressive based on the resources used.
 







