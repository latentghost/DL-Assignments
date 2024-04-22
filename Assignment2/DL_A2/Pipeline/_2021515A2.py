import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader, random_split

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import math

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
# image_dataset_downloader = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=[torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
# )

# # Audio Downloader
# audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
#     root='./data',subset=None,download=True
# )


class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        self.data = torchvision.datasets.CIFAR10(
            root='./data',train=(self.datasplit=='train' or self.datasplit=='val'),download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        )

        if(self.datasplit=='train'):
            self.imagedata,_ = random_split(self.data,[int(0.7*len(self.data)),len(self.data)-int(0.7*len(self.data))])
        elif(self.datasplit=='val'):
            _,self.imagedata = random_split(self.data,[int(0.7*len(self.data)),len(self.data)-int(0.7*len(self.data))])
        else:
            self.imagedata = self.data
        """
        Write your code here
        """

    def __len__(self):
        return len(self.imagedata)
    
    def __getitem__(self,idx):
        image,label = self.imagedata[idx]
        return image,label

    
class ResizeAudio:
    def __init__(self,target_length:int=16000) -> None:
        self.target_length = target_length
    
    def __call__(self,waveform):
        if(waveform.shape[1]<self.target_length):
            pad_size = self.target_length-waveform.shape[1]
            return torch.nn.functional.pad(waveform,(0,pad_size))
        else:
            return torch.nn.functional.interpolate(waveform.unsqueeze(0),(self.target_length,),mode='nearest-exact').squeeze(0)
    

class AudioDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        
        subset = 'training'
        if(self.datasplit=='val'):
            subset = 'validation'
        elif(self.datasplit=='test'):
            subset = 'testing'

        self.audiodata = torchaudio.datasets.SPEECHCOMMANDS(
            root='./data',download=True,subset=subset
        )

        self.labels = ['backward','bed','bird','cat','dog','down','eight',
                       'five','follow','forward','four','go','happy','house',
                       'learn','left','marvin','nine','no','off','on','one',
                       'right','seven','sheila','six','stop','three','tree',
                       'two','up','visual','wow','yes','zero']

        self.label_mapping = self.create_label_mapping()
        """
        Write your code here
        """

    def create_label_mapping(self):
        unique_labels = sorted(set(self.labels))
        label_mapping = {label: index for index, label in enumerate(unique_labels)}
        return label_mapping

    def __len__(self):
        return len(self.audiodata)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.audiodata[idx]
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,new_freq=4800)(waveform)
        waveform = ResizeAudio(target_length=6400)(waveform)
        return waveform,self.label_mapping[label]
    

class ResNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,padding=0,dilation=1,kernel_size=3,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.conv21 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=1,padding=kernel_size-1,dilation=2)
        self.bn22 = nn.BatchNorm2d(out_channels)

        self.conv11 = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn11 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv1d(out_channels,out_channels,kernel_size=kernel_size,stride=1,padding=kernel_size-1,dilation=2)
        self.bn12 = nn.BatchNorm1d(out_channels)

        self.stride = stride
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        if(len(x.shape)==3):
            d = 1
        else:
            d = 2
        identity = x

        # print(x.shape)

        if d==2:
            out = self.conv21(x)
            out = self.bn21(out)
            out = self.relu2(out)
            out = self.conv22(out)
            out = self.bn22(out)
            if out.shape!=identity.shape:
                identity = self.conv21(identity)
                identity = self.bn21(identity)
        else:
            out = self.conv11(x)
            out = self.bn11(out)
            out = self.relu1(out)
            out = self.conv12(out)
            out = self.bn12(out)
            if out.shape!=identity.shape:
                identity = self.conv11(identity)
                identity = self.bn11(identity)
        # print(out.shape)
        
        # print(identity.shape)
        out += identity
        out = self.relu(out)

        # print(out.shape[1])
        return out
    __call__ = forward


class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.b2 = ResNetBlock(3,32,2,2,2)
        self.b1 = ResNetBlock(1,32,2,2,2)
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.blocks = nn.Sequential(
            ResNetBlock(32,16,1,2,2),
            ResNetBlock(16,32,1,1,2),
            ResNetBlock(32,16,1,2,2),
            ResNetBlock(16,8,1,1,2),
            ResNetBlock(8,16,1,1,2),
            ResNetBlock(16,8,1,2,2),
            ResNetBlock(8,16,1,1,1),
            ResNetBlock(16,24,1,2,2),
            ResNetBlock(24,16,1,2,2),
            ResNetBlock(16,8,1,2,2),
            ResNetBlock(8,16,1,2,2),
            ResNetBlock(16,8,1,2,2),
            ResNetBlock(8,24,1,2,2),
            ResNetBlock(24,12,1,1,2),
            ResNetBlock(12,24,1,1,2),
            ResNetBlock(24,20,1,1,1)
        )
        self.bl2 = ResNetBlock(20,10,1,1,1)
        self.bl1 = ResNetBlock(20,35,1,1,1)
        """
        Write your code here
        """

    def forward(self, x):
        if(len(x.shape)==3):
            x = self.b1(x)
            x = self.blocks(x)
            x = self.bl1(x)
            # print(x.shape)
            x = self.pool1(x)
            # print(x.shape)
            return x
        else:
            x = self.b2(x)
            x = self.blocks(x)
            x = self.bl2(x)
            # print(x.shape)
            x = self.pool2(x)
            # print(x.shape)
            return x

    __call__ = forward


class VGGBlock(nn.Module):
    def __init__(self,in_channels,out_channels,n_layers,kernel_size=2,stride=1,padding=1,dilation=1,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.conv2 = nn.ModuleList()
        self.conv21 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.conv2.append(self.conv21)
        self.conv2.append(self.bn21)

        for i in range(n_layers-1):
            setattr(self,f'conv2{i+2}',nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation))
            setattr(self,f'bn2{i+2}',nn.BatchNorm2d(out_channels))
            self.conv2.append(getattr(self,f'conv2{i+2}'))
            self.conv2.append(getattr(self,f'bn2{i+2}'))
            
        self.conv1 = nn.ModuleList()
        self.conv11 = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn11 = nn.BatchNorm1d(out_channels)
        self.conv1.append(self.conv11)
        self.conv1.append(self.bn11)

        for i in range(n_layers-1):
            setattr(self,f'conv1{i+2}',nn.Conv1d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation))
            setattr(self,f'bn1{i+2}',nn.BatchNorm1d(out_channels))
            self.conv1.append(getattr(self,f'conv1{i+2}'))
            self.conv1.append(getattr(self,f'bn1{i+2}'))

    def forward(self,x):
        if len(x.shape)==3:
            for i,conv in enumerate(self.conv1):
                x = conv(x)
                if(i%2==1):
                    x = nn.ReLU(inplace=True)(x)
            x = nn.MaxPool1d(kernel_size=2,stride=1,padding=0)(x)
        else:
            for i,conv in enumerate(self.conv2):
                x = conv(x)
                if(i%2==1):
                    x = nn.ReLU(inplace=True)(x)
            x = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)(x)
        return x
    __call__ = forward

        
class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.block11 = VGGBlock(1,64,2,kernel_size=1,stride=2,padding=2,dilation=2)
        self.block21 = VGGBlock(3,64,2,kernel_size=1,stride=2,padding=2,dilation=2)
        # 10 x 10
        self.block2 = VGGBlock(64,42,2,kernel_size=2,stride=1,padding=2,dilation=2)
        # 6 x 6
        self.block3 = VGGBlock(42,28,3,kernel_size=3,stride=1,padding=1,dilation=1)
        # 17 x 17
        self.block4 = VGGBlock(28,19,3,kernel_size=4,stride=1,padding=1,dilation=1)
        # 7 x 7
        self.block25 = VGGBlock(19,13,3,kernel_size=5,stride=1,padding=1,dilation=1)
        # 2 x 2
        self.block15 = VGGBlock(19,12,3,kernel_size=5,stride=2,padding=1,dilation=1)
        
        self.fc2 = nn.Sequential(
            nn.Linear(13,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,10)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(12*198,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,35)
        )

        """
        Write your code here
        """

    def forward(self, x):
        if(len(x.shape)==3):
            x = self.block11(x)
        else:
            x = self.block21(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if(len(x.shape)==3):
            x = self.block15(x)
            # print(x.shape)
            x = x.view(x.size(0),-1)
            x = self.fc1(x)
        else:
            x = self.block25(x)
            # print(x.shape)
            x = x.view(x.size(0),-1)
            x = self.fc2(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x
    

class CNA(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self,x):
        if(len(x.shape)==3):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
        return x
    

class InceptionBlock(nn.Module):
    def __init__(self,in_channels,out_channels,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        out_channels = out_channels//4

        self.cna1 = CNA(in_channels,out_channels,1,2,0,1)
        self.cna21 = CNA(in_channels,out_channels,3,1,1,1)
        self.cna22 = CNA(out_channels,out_channels,5,2,2,1)
        self.cna31 = CNA(in_channels,out_channels,3,1,1,1)
        self.cna32 = CNA(out_channels,out_channels,5,2,2,1)
        self.cna4 = CNA(in_channels,out_channels,1,2,0,1)
        self.cna4_1 = nn.MaxPool1d(kernel_size=3,stride=1,padding=1,dilation=1)
        self.cna4_2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1,dilation=1)

    def forward(self, x):
        x1 = self.cna1(x)
        x2 = self.cna21(x)
        x2 = self.cna22(x2)
        x3 = self.cna31(x)
        x3 = self.cna32(x3)
        x4 = self.cna4(x)
        if(len(x.shape)==3):
            x4 = self.cna4_1(x1)
        else:
            x4 = self.cna4_2(x1)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x
        
class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        self.block21 = InceptionBlock(3,8)
        self.block11 = InceptionBlock(1,8)
        self.block2 = InceptionBlock(8,16)
        self.block3 = InceptionBlock(16,32)
        self.block4 = InceptionBlock(32,4)
        
        self.dense1 = nn.Sequential(
            nn.Linear(4*4*100,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,35)
        )
        self.dense2 = nn.Linear(4*4,10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if(len(x.shape)==3):
            x = self.block11(x)
        else:
            x = self.block21(x)
        
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if(len(x.shape)==3):
            x = x.view(x.size(0),-1)
            x = self.dense1(x)
        else:
            x = x.view(x.size(0),-1)
            x = self.dense2(x)
        x = self.logsoftmax(x)
        
        return x
        

class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        self.res11 = ResNetBlock(1,32,1,0,1,1)
        self.res21 = ResNetBlock(3,32,1,0,1,1)
        self.res2 = ResNetBlock(32,21,1,2,2,2)
        self.inc1 = InceptionBlock(21,16)
        self.inc2 = InceptionBlock(16,12)
        self.res3 = ResNetBlock(12,8,1,3,1,3)
        self.inc3 = InceptionBlock(8,8)
        self.res4 = ResNetBlock(8,4,1,3,1,4)
        self.inc4 = InceptionBlock(4,4)
        self.res5 = ResNetBlock(4,3,1,2,1,5)
        self.inc5 = InceptionBlock(3,4)
        
        self.fc21 = nn.Linear(4*3*3,16)
        self.fc11 = nn.Linear(4*202,128)
        self.fc22 = nn.Linear(16,10)
        self.fc12 = nn.Linear(128,64)
        self.fc13 = nn.Linear(64,35)

    def forward(self, x):
        if (len(x.shape)==3):
            x = self.res11(x)
        else:
            x = self.res21(x)
        x = self.res2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.res3(x)
        x = self.inc3(x)
        x = self.res4(x)
        x = self.inc4(x)
        x = self.res5(x)
        x = self.inc5(x)
        
        if(len(x.shape)==3):
            x = x.view(x.size(0),-1)
            x = self.fc11(x)
            x = nn.ReLU(inplace=True)(x)
            x = self.fc12(x)
            x = nn.ReLU(inplace=True)(x)
            x = self.fc13(x)
        else:
            x = x.view(x.size(0),-1)
            x = self.fc21(x)
            x = nn.ReLU(inplace=True)(x)
            x = self.fc22(x)
        return nn.LogSoftmax(dim=1)(x)


def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1,3,1,1)

    best_acc = 0
    best_loss = math.inf
    
    # Write your code here
    for epoch in range(EPOCH):
        network.train()
        correct = 0
        n_samples = 0
        total_loss = 0.0

        for idx,batch in enumerate(dataloader):

            data,label = batch

            ext = "image"
            if(len(data.shape)==3):
                ext = "audio"

            if(len(data.shape)==4):
                data = (data-mean)/std
            
            # print(data.shape)

            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = network(data)
            output = output.reshape(output.shape[0],-1)
            loss = criterion(output,label)

            loss.backward()
            optimizer.step()

            correct += (output.argmax(dim=1) == label).sum().item()
            n_samples += label.size(0)
            total_loss += loss.item()

            # print(f"Batch {idx} done")

        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            total_loss/len(dataloader),
            correct/n_samples
        ))

        if(total_loss<best_loss):
            if(correct/n_samples>best_acc):
                best_loss = total_loss
                best_acc = correct/n_samples
                save_dict = {
                    'model': network.state_dict(),
                    'best_loss': best_loss,
                    'best_acc': best_acc
                }
                torch.save(save_dict,f'best_model_{network.__class__.__name__}_{ext}.pth')

        if(best_acc>0.8):
            break
        
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda") if gpu == "T" else torch.device("cpu")
    network = network.to(device)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1,3,1,1)

    best_acc = None
    best_loss = None
    
    # Write your code here
    for epoch in range(EPOCH):
        network.train()
        correct = 0
        n_samples = 0
        total_loss = 0.0

        for batch in dataloader:
            data,label = batch

            ext = "image"
            if(len(data.shape)==3):
                ext = "audio"

            if(len(data.shape)==4):
                image_dict = torch.load(f'best_model_{network.__class__.__name__}_image.pth')
                data = (data-mean)/std
                if(best_loss is None):
                    network.load_state_dict(image_dict['model'])
                    best_loss = image_dict['best_loss']
                    best_acc = image_dict['best_acc']
                    network.train()
            elif(best_loss is None):
                audio_dict = torch.load(f'best_model_{network.__class__.__name__}_audio.pth')
                network.load_state_dict(audio_dict['model'])
                best_loss = audio_dict['best_loss']
                best_acc = audio_dict['best_acc']
                network.train()

            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = network(data)
            output = output.reshape(output.shape[0],-1)
            loss = criterion(output,label)

            loss.backward()
            optimizer.step()

            correct += (output.argmax(dim=1) == label).sum().item()
            n_samples += label.size(0)
            total_loss += loss.item()

        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            total_loss/len(dataloader),
            correct/n_samples
        ))

        if(total_loss<best_loss):
            best_loss = total_loss
            if(correct/n_samples>best_acc):
                best_acc = correct/n_samples
                save_dict = {
                    'model': network.state_dict(),
                    'best_loss': best_loss,
                    'best_acc': best_acc
                }
                torch.save(save_dict,f'best_model_{network.__class__.__name__}_{ext}.pth')

        if(best_acc>0.85):
            break
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda") if gpu == "T" else torch.device("cpu")
    network = network.to(device)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1,3,1,1)
    
    # Write your code here
    for epoch in range(EPOCH):
        pass

    network.eval()
    correct = 0
    n_samples = 0

    loaded = False

    for batch in dataloader:
        data,label = batch
        
        if(not loaded):
            if(len(data.shape)==4):
                image_dict = torch.load(f'best_model_{network.__class__.__name__}_image.pth')
                data = (data-mean)/std
                network.load_state_dict(image_dict['model'])
                network.eval()
            else:
                audio_dict = torch.load(f'best_model_{network.__class__.__name__}_audio.pth')
                network.load_state_dict(audio_dict['model'])
                network.eval()
            loaded = True

        data = data.to(device)
        label = label.to(device)

        output = network(data)
        output = output.reshape(output.shape[0],-1)

        correct += (output.argmax(dim=1) == label).sum().item()
        n_samples += label.size(0)

    print("[Accuracy: {}]".format(
        correct/n_samples
    ))
    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """