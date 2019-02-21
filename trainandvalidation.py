from __future__ import print_function
import argparse
from fastai.vision import *
from glob import glob
from matplotlib import pyplot as plt
import torch.optim as optim
import numpy as np


# Creating our own customized dataset class in pytorch
 
class CocoDdataset(Dataset):
    def __init__(self,path):
        X=()
        Y=()
        lst = os.listdir(path)
        it=0
        for i in lst:
            array = np.load(path+'%s'%i)
            x = torch.from_numpy((array[0].astype(float)-127.5)/127.5)
            X = X+(x,)
            y = torch.from_numpy(array[1].astype(float) / 32.)
            Y = Y+(y,)
            it+=1
        self.len = it
        self.X_data = X
        self.Y_data = Y
    def __getitem__(self,index):
        return self.X_data[index], self.Y_data[index] 
    def __len__(self):
        return self.len
    



train_path = '/home/jupyter/.fastai/data/train2017processed/'
validation_path = '/home/jupyter/.fastai/data/val2017processed/'
test_path = '/home/jupyter/.fastai/data/test2017processed/'

TrainingData = CocoDdataset(train_path)
ValidationData = CocoDdataset(validation_path)
TestData = CocoDdataset(test_path)

TrainLoader = DataLoader(TrainingData,batch_size=64)
ValidationLoader = DataLoader(ValidationData,batch_size=64)
TestLoader = DataLoader(TestData,batch_size=64)
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
#as in paper 
num_samples = 118287
total_iteration = 90000
batch_size = 64
steps_per_epoch = num_samples / batch_size
epochs = int(total_iteration / steps_per_epoch)
model = Model().to(device)
optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
checkpoint = torch.load('model1.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
for epoch in range(2):
    
    for i, (images, target) in enumerate(TrainLoader):
        images = images.to(device)
        target = target.to(device)
        images = images.view(-1,2,128,128)
        images = images.float()
        target = target.float()
        outputs = model(images)
        loss = criterion(outputs, target.view(-1,8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 512 == 0:
            print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\Mean Squared Error: {:.6f}'.format(
                epoch+1,epochs, i * len(images), len(TrainLoader),
                100. * i / len(TrainLoader), loss))

state = {'epoch': epochs, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict() }
torch.save(state, 'model.pth')

model.eval()
for epoch in range(epochs):
    with torch.no_grad():
        
        for i,(images, target) in enumerate(ValidationLoader):

            images = images.to(device)
            target = target.to(device)
            images = images.view(-1,2,128,128)
            images = images.float()
            target = target.float()
            outputs = model(images)
        loss = criterion(outputs, target.view(-1,8))
            if (i+1) % 512 == 0:
                print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\Mean Squared Error: {:.6f}'.format(
                epoch+1,epochs, i * len(images), len(ValidationLoader),
                100. * i / len(ValidationLoader), loss))
