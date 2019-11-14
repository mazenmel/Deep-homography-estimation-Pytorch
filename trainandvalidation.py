from fastai.vision import *
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Create a customized dataset class in pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
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
    



train_path = '/home/jupyter/train2017/train2017processed/'
validation_path = '/home/jupyter/val2017/val2017processed/'
test_path = '/home/jupyter/test2017/test2017processed/'

TrainingData = CocoDdataset(train_path)
ValidationData = CocoDdataset(validation_path)
TestData = CocoDdataset(test_path)

# Training

batch_size = 64
TrainLoader = DataLoader(TrainingData,batch_size)
ValidationLoader = DataLoader(ValidationData,batch_size)
TestLoader = DataLoader(TestData,batch_size)
criterion = nn.MSELoss()
num_samples = 118287
total_iteration = 90000
steps_per_epoch = num_samples / batch_size
epochs = int(total_iteration / steps_per_epoch)
model = Model().to(device)
summary(model,(2,128,128))
optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
for epoch in range(epochs):
    
    for i, (images, target) in enumerate(TrainLoader):
        optimizer.zero_grad()
        images = images.to(device); target = target.to(device)
        images = images.permute(0,3,1,2).float(); target = target.float()
        outputs = model(images)
        loss = criterion(outputs, target.view(-1,8))
        loss.backward()
        optimizer.step()
        if (i+1) % len(TrainLoader) == 0:
            print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\Mean Squared Error: {:.6f}'.format(
                epoch+1,epochs, i , len(TrainLoader),
                100. * i / len(TrainLoader), loss))

state = {'epoch': epochs, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict() }
torch.save(state, 'DeepHomographyEstimation.pth')

# Evaluate on validation set

model.eval()
with torch.no_grad():
    for i,(images, target) in enumerate(ValidationLoader):
        images = images.to(device)
        target = target.to(device)
        images = images.permute(0,3,1,2).float()
        target = target.float()
        outputs = model(images)
        loss = criterion(outputs, target.view(-1,8))
        print('\Mean Squared Error: {:.6f}'.format(loss))