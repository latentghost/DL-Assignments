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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
        self.to_grayscale = Grayscale()
        self.aug = os.path.join(root, "aug/")
        self.clean = os.path.join(root, "clean/")
        self.aug_paths = (os.listdir(self.aug)[:-3000] if (not val) else os.listdir(self.aug)[-3000:])
        self.aug_len = len(self.aug_paths)
        self.clean_len = len(os.listdir(self.clean))

        self.targets = [[] for _ in range(10)]
        for file in os.listdir(self.clean):
            if file.endswith(".png"):
                label = int(file.split("_")[2].split(".")[0])
                image_path = os.path.join(self.clean, file)
                image = (torchvision.io.read_image(image_path)).float().div_(255.0)
                self.targets[label].append(image)

    def __getitem__(self, index):
        img_path = self.aug_paths[index]
        img = (torchvision.io.read_image(os.path.join(self.aug,img_path))).float().div_(255.0)
        idx = int(img_path.split("_")[1])
        label = int(img_path.split("_")[2].split(".")[0])

        if img.shape[0] == 3:
            img = self.to_grayscale(img)

        if idx < self.clean_len:
            target = (torchvision.io.read_image(os.path.join(self.clean,f"clean_{idx}_{label}.png"))).float().div_(255.0)
        else:
            random_els = random.sample(self.targets[label][1:], 10)

            best_target = (random_els[0])
            best_score = ssim(img, best_target)

            for candidate in random_els:
                ssim_score = ssim(img, candidate)
                if ssim_score > best_score:
                    best_score = ssim_score
                    best_target = candidate
            target = best_target

        img = img.to(torch.float32)
        target = target.to(torch.float32)
        
        return img, target
    
    def __len__(self):
        return self.aug_len


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
        )
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self,in_channels=1,out_channels=8,latent_dim=10):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.block1 = EncoderBlock(2, 4, 3, 2, 2)
        self.block2 = EncoderBlock(4, 4, 3, 2, 1)
        self.block3 = EncoderBlock(4, out_channels, 2, 1, 0)
        self.block4 = EncoderBlock(out_channels, out_channels, 3, 1, 1)
        self.fc_mu = nn.Linear(out_channels*3*3,latent_dim)
        self.fc_logvar = nn.Linear(out_channels*3*3,latent_dim)

    def forward(self, x, VAE=False):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if VAE:
            x_ = x.view(x.shape[0],-1)
            mu = self.fc_mu(x_)
            logvar = self.fc_logvar(x_)
            return x, mu, logvar
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.upsample(identity)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self, in_channels=8, out_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.block1 = DecoderBlock(8, 4, 3, 2, 1)
        self.block2 = DecoderBlock(4, 4, 3, 2, 0)
        self.block3 = DecoderBlock(4, 2, 4, 1, 0)
        self.block4 = DecoderBlock(2, 1, 3, 1, 1)
        self.conv3 = nn.ConvTranspose2d(1, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
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
        self.loss = nn.MSELoss()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        logits = logits.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        loss = self.loss(logits, targets)
        return loss


class VAELossFn(nn.Module):
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    def __init__(self, reduction:str = 'mean'):
        super(VAELossFn, self).__init__()
        self.reconstruct_loss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, logits, targets, mu, logvar):
        logits = logits.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        reconstruct_loss = self.reconstruct_loss(logits, targets)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruct_loss + KLD


def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    return iter(list(E.parameters()) + list(D.parameters()))


def plot_tsne(logits,epoch,VAE=False):
    logits = logits.view(logits.shape[0],-1).numpy()
    pca = PCA(n_components=30)
    logits = pca.fit_transform(logits)
    tsne = TSNE(n_components=3)
    tsne_logits = tsne.fit_transform(logits)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_logits[:,0], tsne_logits[:,1], tsne_logits[:,2])
    plt.savefig("tsne_plots/{}AE_epoch_{}.png".format(("V" if VAE else ""),epoch))
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

            epoch_loss = []
            epoch_sim = []

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.encoder(data)
                output = self.decoder(logits)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,logits.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch+1,loss,similarity))

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))
            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch)
        
        save_checkpoint(self.encoder,self.decoder,"checkpoints/AE_trained.pth")


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
        self.gpu = gpu
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self,optimizer,loss_fn,VAE=True):
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])

            epoch_loss = []
            epoch_sim = []

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits,mu,logvar = self.encoder(data,VAE)
                output = self.decoder(logits)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,logits.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch+1,loss,similarity))

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,VAE)

        save_checkpoint(self.encoder,self.decoder,"checkpoints/VAE_trained.pth")


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
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpoints/AE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample_img = torchvision.io.read_image(sample).float().div_(255.0).to(self.device)
        original_img = torchvision.io.read_image(original).float().div_(255.0).to(self.device)
        logits = self.encoder(sample_img)
        output = self.decoder(logits)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return peak_signal_to_noise_ratio(output, original_img)
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
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpoints/VAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample_img = torchvision.io.read_image(sample).float().div_(255.0).to(self.device)
        original_img = torchvision.io.read_image(original).float().div_(255.0).to(self.device)
        logits = self.encoder(sample_img,VAE=True)
        output = self.decoder(logits)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return peak_signal_to_noise_ratio(output, original_img)
        else:
            raise Exception("Invalid type. Use 'SSIM' or 'PSNR'.")


class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass


class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass


class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def ssim(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    channel = 1
    window_size = 11
    K = [0.05, 0.05]
    C1 = K[0]**2
    C2 = K[1]**2

    sigma = 1.5 
    gauss = [torch.exp(torch.tensor(-(x - window_size//2)**2/(2*sigma**2))) for x in range(window_size)]
    gauss = torch.tensor(gauss)
    window1d = (gauss/gauss.sum()).unsqueeze(1)
    window2d = window1d.mm(window1d.transpose(0,1)).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window2d.expand(channel, 1, window_size, window_size).contiguous())
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu12

    ssim_score = ((2*mu12 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return torch.clamp(ssim_score.mean(),min=0,max=1)