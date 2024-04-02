import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Grayscale
from torch.autograd import Variable
import os
import random
from EncDec import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


class AlteredMNIST(torch.utils.data.Dataset):
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Write code to load Dataset
    """
    def __init__(self, root="./Data/", val=False):
        super(AlteredMNIST, self).__init__()
        self.root = root
        self.aug = os.path.join(root, "aug/")
        self.clean = os.path.join(root, "clean/")
        self.aug_paths = (os.listdir(self.aug)[:-3000] if not val else os.listdir(self.aug)[-3000:])
        self.aug_len = len(self.aug_paths)
        self.clean_len = len(os.listdir(self.clean))

        self.w_r = nn.Parameter(torch.tensor(0.2126),requires_grad=False)
        self.w_b = nn.Parameter(torch.tensor(0.7152),requires_grad=False)
        self.w_g = nn.Parameter(torch.tensor(0.0722),requires_grad=False)

        self.targets = [[] for _ in range(10)]
        for file in os.listdir(self.clean):
            if file.endswith(".png"):
                label = int(file.split("_")[2].split(".")[0])
                image_path = os.path.join(self.clean, file)
                self.targets[label].append(image_path)

    def __getitem__(self, index):
        img_path = self.aug_paths[index]
        img = (torchvision.io.read_image(os.path.join(self.aug,img_path))).float()
        idx = int(img_path.split("_")[1])
        label = int(img_path.split("_")[2].split(".")[0])

        if img.shape[0] == 3:
            img = self.to_grayscale(img).unsqueeze(0)

        if idx < self.clean_len:
            target = (torchvision.io.read_image(os.path.join(self.clean,f"clean_{idx}_{label}.png"))).float()
        else:
            random_els = random.sample(self.targets[label], 10)
            best_target = random_els[0]
            best_score = 0.5*(ssim(img, torchvision.io.read_image(best_target).float())) + 0.5*(psnr(img, torchvision.io.read_image(best_target).float()))

            for candidate in random_els[1:]:
                psnr_score = 0.5*(ssim(img, torchvision.io.read_image(candidate).float())) + 0.5*(psnr(img, torchvision.io.read_image(candidate).float()))
                if psnr_score > best_score:
                    best_score = psnr_score
                    best_target = candidate
            target = torchvision.io.read_image(best_target).float()

        img = img.to(torch.float32).div_(255.)
        target = target.to(torch.float32).div_(255.)
        return img, target, label
    
    def __len__(self):
        return self.aug_len
    
    def to_grayscale(self, img):
        return self.w_r*img[0] + self.w_g*img[1] + self.w_b*img[2]
    

def one_hot(x,num_classes):
    out = torch.zeros(x.size(0),num_classes)
    for i,l in enumerate(x):
        out[i][l] = 1
    return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.SELU()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        # out = self.relu(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self,in_channels=1,out_channels=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.block1 = EncoderBlock(2, 4, 3, 2, 2)
        self.block2 = EncoderBlock(4, 8, 3, 2, 1)
        self.block3 = EncoderBlock(8, out_channels, 2, 1, 0)
        self.block4 = EncoderBlock(out_channels, out_channels, 3, 1, 1)
        self.fc_mu = nn.Linear(out_channels*3*3,out_channels)
        self.fc_logvar = nn.Linear(out_channels*3*3,out_channels)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.fc1 = nn.Linear(out_channels*3*3 + 10, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.fc3 = nn.Linear(64, out_channels)

        self.mu = None
        self.logvar = None
        self.z_mu = None
        self.z_logvar = None

    def forward(self, x, label=None, VAE=False, CVAE=False):
        x = self.conv1(x)
        x = self.selu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if VAE or CVAE:
            x_ = x.view(x.shape[0],-1)
            mu = self.fc_mu(x_)
            logvar = self.fc_logvar(x_)
            if CVAE:
                targets = one_hot(label,10).to(x_.device)
                inputs = torch.cat((x_,targets),dim=1)
                h1 = self.elu(self.fc1(inputs))
                z_mu = self.fc2(h1)
                z_logvar = self.fc3(h1)
                self.z_mu = z_mu
                self.z_logvar = z_logvar
                return x, self.reparametrize(VAE,CVAE), z_mu, z_logvar
            self.mu = mu
            self.logvar = logvar
            return x, self.reparametrize(VAE,CVAE), mu, logvar
        return x
    
    def reparametrize(self, VAE=False, CVAE=False):
        if VAE:
            std = torch.exp_(0.5*self.logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(self.mu)
        elif CVAE:
            std = torch.exp_(0.5*self.z_logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(self.z_mu)
        else:
            return None
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            # nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.conv4(out)
        # out = self.relu(out)
        out += self.upsample(identity)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self, in_channels=16, out_channels=1):
        super(Decoder, self).__init__()
        self.fc_cvae = nn.Linear(in_channels+10, in_channels*3*3)
        self.fc_vae = nn.Linear(in_channels, in_channels*3*3)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.block1 = DecoderBlock(in_channels, 8, 3, 2, 1)
        self.block2 = DecoderBlock(8, 4, 3, 2, 0)
        self.block3 = DecoderBlock(4, 2, 4, 1, 0)
        self.block4 = DecoderBlock(2, 2, 3, 1, 1)
        self.conv3 = nn.ConvTranspose2d(2, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, label=None, VAE=False, CVAE=False):
        if VAE or CVAE:
            if CVAE:
                targets = one_hot(label,10).to(x.device)
                x = torch.cat((x,targets),dim=1)
                x = self.elu(self.fc_cvae(x))
            else:
                x = self.elu(self.fc_vae(x))
            x = x.view(x.size(0),-1,3,3)
        x = self.conv1(x)
        x = self.selu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x
    

class AELossFn(nn.Module):
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self, reduction:str = 'mean'):
        super(AELossFn, self).__init__()
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        logits = logits.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        loss = 0.2*(self.bceloss(logits,targets)) + 0.8*(self.mseloss(logits, targets))
        return loss


class VAELossFn(nn.Module):
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """  
    def __init__(self, reduction:str = 'mean'):
        super(VAELossFn, self).__init__()
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, logits, targets, mu, logvar):
        logits = logits.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        reconstruct_loss = 0.2*(self.bceloss(logits, targets)) + 0.8*(self.mseloss(logits, targets))
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruct_loss + KLD


def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    return (list(E.parameters()) + list(D.parameters()))


def plot_tsne(logits,epoch,labels=None,VAE=False,CVAE=False):
    logits = logits.view(logits.shape[0],-1).numpy()
    tsne = TSNE(n_components=3)
    tsne_logits = tsne.fit_transform(logits)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_logits[:,0], tsne_logits[:,1], tsne_logits[:,2], c=labels, cmap='tab10')
    plt.savefig("{}AE_epoch_{}.png".format(("V" if VAE else "CV" if CVAE else ""),epoch))
    plt.close()


def save_checkpoint(encoder, decoder, path):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict()
    }, path)

def load_checkpoint(encoder, decoder, path):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return encoder, decoder


class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self, data_loader:torch.utils.data.DataLoader, encoder:Encoder, decoder:Decoder, loss_fn, optimizer, gpu="F"):
        self.data_loader = data_loader
        self.encoder = encoder
        self.decoder = decoder
        self.gpu = gpu
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self,optimizer,loss_fn):
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, labels) in enumerate(self.data_loader):
                data, target, labels = data.to(self.device), target.to(self.device), labels.to(self.device)
                logits = self.encoder(data)
                output = self.decoder(logits)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,logits.clone().cpu().detach()),dim=0)
                    labels_tsne = torch.cat((labels_tsne,labels.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))

                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))

            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, _) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.encoder(data)
                output = self.decoder(logits)

                loss = loss_fn(output, target)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,labels_tsne)
            
        save_checkpoint(self.encoder,self.decoder,"checkpointAE_trained.pth")


class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self, data_loader:torch.utils.data.DataLoader, encoder:Encoder, decoder:Decoder, loss_fn, optimizer, gpu="F"):
        self.data_loader = data_loader
        self.encoder = Encoder()
        self.decoder = Decoder()
        optimizer.param_groups[0]['params'] = ParameterSelector(self.encoder,self.decoder)
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self,optimizer,loss_fn):
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, labels) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits,reparam,mu,logvar = self.encoder(data,VAE=True)
                output = self.decoder(reparam,VAE=True)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,logits.clone().cpu().detach()),dim=0)
                    labels_tsne = torch.cat((labels_tsne,labels.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))

            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, _) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                _,reparam,mu,logvar = self.encoder(data,VAE=True)
                output = self.decoder(reparam,VAE=True)

                loss = loss_fn(output, target, mu, logvar)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,labels_tsne,VAE=True)

        save_checkpoint(self.encoder,self.decoder,"checkpointVAE_trained.pth")


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu=False):
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample_img = torchvision.io.read_image(sample).float().div_(255.0).to(self.device)
        original_img = torchvision.io.read_image(original).float().div_(255.0).to(self.device)
        sample_img = sample_img.unsqueeze(0).to(self.device)
        original_img = original_img.unsqueeze(0).to(self.device)
        logits = self.encoder(sample_img)
        output = self.decoder(logits)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return psnr(output, original_img)
        else:
            raise Exception("Invalid type. Use 'SSIM' or 'PSNR'.")
        

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu=False):
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointVAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample_img = torchvision.io.read_image(sample).float().div_(255.0).to(self.device)
        original_img = torchvision.io.read_image(original).float().div_(255.0).to(self.device)
        sample_img = sample_img.unsqueeze(0).to(self.device)
        original_img = original_img.unsqueeze(0).to(self.device)
        _,reparam,_,_ = self.encoder(sample_img,VAE=True)
        output = self.decoder(reparam,VAE=True)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return psnr(output, original_img)
        else:
            raise Exception("Invalid type. Use 'SSIM' or 'PSNR'.")


class CVAELossFn(nn.Module):
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    def __init__(self, reduction:str = 'mean'):
        super(CVAELossFn, self).__init__()
        self.reconstruct_loss = nn.BCEWithLogitsLoss()
        self.reduction = reduction

    def forward(self, logits, targets, mu, logvar, label):
        logits = logits.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        reconstruct_loss = self.reconstruct_loss(logits, targets)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return torch.mean((reconstruct_loss + KLD)*label)


class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    def __init__(self, data_loader:torch.utils.data.DataLoader, encoder:Encoder, decoder:Decoder, loss_fn, optimizer):
        self.data_loader = data_loader
        self.encoder = Encoder()
        self.decoder = Decoder()
        optimizer.param_groups[0]['params'] = ParameterSelector(self.encoder,self.decoder)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self, optimizer, loss_fn):
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, label) in enumerate(self.data_loader):
                data, target, label = (data.to(self.device),
                                       target.to(self.device),
                                       label.to(self.device))
                logits,reparam,mu,logvar = self.encoder(data,label,CVAE=True)
                output = self.decoder(reparam,label,CVAE=True)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,logits.clone().cpu().detach()),dim=0)
                    labels_tsne = torch.cat((labels_tsne,label.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target, mu, logvar, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
            
            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, label) in enumerate(self.data_loader):
                data, target, label = data.to(self.device), target.to(self.device), label.to(self.device)
                _,reparam,mu,logvar = self.encoder(data,label,CVAE=True)
                output = self.decoder(reparam,label,CVAE=True)

                loss = loss_fn(output, target, mu, logvar, label)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,labels_tsne,CVAE=True)
        
        save_checkpoint(self.encoder,self.decoder,"checkpointCVAE_trained.pth")


class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    def __init__(self, gpu=False):
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointCVAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def save_image(self, digit, save_path):
        z = self.encoder.reparametrize(VAE=False,CVAE=True)
        output = self.decoder(z,digit,CVAE=True)
        torchvision.io.write_png(output.squeeze(0),os.path.join(save_path,f"label_{digit}.png"))


def psnr(img1, img2, max_val:float=255.0):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    diff = img1.to(torch.float64).sub_(img2.to(torch.float64))
    mse = torch.mean(diff ** 2)
    if mse==0: return float("inf")
    return 20 * torch.log10(max_val/torch.sqrt(mse)).item()


def ssim(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    channel = 1
    window_size = 11
    K = [0.03, 0.03]
    C1 = K[0]**2
    C2 = K[1]**2

    sigma = 1.
    gauss = [torch.exp_(torch.tensor(-(x - window_size//2)**2/(2*sigma**2))) for x in range(window_size)]
    gauss = torch.tensor(gauss)
    window1d = (gauss/gauss.sum()).unsqueeze(1)
    window2d = window1d.mm(window1d.transpose(0,1)).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window2d.expand(channel, 1, window_size, window_size).contiguous())
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow_(2)
    mu2_sq = mu2.pow_(2)
    mu12 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu12

    numerator = (2*mu12 + C1)*(2*sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)
    ssim_score = numerator/denominator
    return torch.clamp(ssim_score.mean(),min=0,max=1)