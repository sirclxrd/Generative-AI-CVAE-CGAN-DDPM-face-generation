import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as tforms
from torchvision.datasets import CelebA
import numpy as np
import PIL
from matplotlib import pyplot as plt
import sys
import os
import time
from torch.nn.functional import binary_cross_entropy


device = torch.device('cuda')
print(device)
LATENT_SIZE=256
COND_SHAPE=(3,)

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.cond_features = COND_SHAPE[0] + 2
        self.latent_size   = latent_size
        self.img_channels  = 3   # RGB
        self.generator     = self.build_generator()

    def build_generator(self):
        model = nn.Sequential()
        prev = 1024
        size =  4 
        model.append(nn.Linear(self.latent_size + self.cond_features, prev * size * size))
        model.append(nn.BatchNorm1d(prev * size * size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))  
        for k in [512 ,256, 128, 64]:
            model.append(nn.ConvTranspose2d(prev, k, kernel_size=4, stride=2, padding=1))
            model.append(nn.BatchNorm2d(k))
            model.append(nn.ReLU())
            prev = k
        model.append(nn.Conv2d(prev, prev, kernel_size=3, padding=1))  
        model.append(nn.BatchNorm2d(prev))
        model.append(nn.ReLU())

        model.append(nn.Conv2d(prev, self.img_channels, kernel_size=3, padding=1))
        model.append(nn.Tanh()) 
        return model

    def forward(self, z, c):
        c= torch.cat([c, c[:, 2].view(-1, 1), c[:, 2].view(-1, 1)], dim=1)
        zc = torch.cat((z, c), dim=1)          
        return self.generator(zc)             


# Esempio di istanziazione:
gen_model     = Generator(LATENT_SIZE).to(device=device)
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0003)



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_features = COND_SHAPE[0]  
        self.discriminator = self.build_discriminator()

    def build_discriminator(self):
        model = nn.Sequential()
        prev_channels = 3 + self.cond_features+ 2  
        channels = [ 64, 128, 256, 512, 1024]

        for k in channels:
            model.append(nn.Conv2d(prev_channels, k, kernel_size=4, stride=2, padding=1))
            model.append(nn.LeakyReLU(0.2))
            prev_channels = k

        model.append(nn.Flatten())
        model.append(nn.Linear(1024 * 2 * 2, 1))
        model.append(nn.Sigmoid())  
        return model

    def forward(self, x, c):
        batch_size, _, H, W = x.shape
        cc = c.view(batch_size, self.cond_features, 1, 1).expand(-1, -1, H, W)
        cond3 = c[:, 2].view(-1, 1, 1, 1).expand(-1, 1, H, W).to(device)  
        cond3_extra = c[:, 2].view(-1, 1, 1, 1).expand(-1, 1, H, W).to(device)  
        xc = torch.cat([x, cc, cond3, cond3_extra], dim=1)
        return self.discriminator(xc)


disc_model     = Discriminator().to(device=device)
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.0002)
LABEL_SMOOTHING = 0.1


def disc_loss_function(d_true, d_synth):
    t_true = torch.ones_like(d_true) - LABEL_SMOOTHING 
    t_synth = torch.zeros_like(d_synth) + LABEL_SMOOTHING
    return binary_cross_entropy(d_true, t_true) + binary_cross_entropy(d_synth, t_synth)

def gen_loss_function(d_synth):
    t_synth = torch.ones_like(d_synth) 
    return binary_cross_entropy(d_synth, t_synth)




def training_epoch(dataloader):
    selected_indices = [15, 20, 24]
    gen_model.train()
    disc_model.train()
    sum_gloss = 0.0
    sum_dloss = 0.0
    sum_dtrue = 0.0
    sum_dsynth = 0.0
    batches = 0 
    for x_true, cls in dataloader:
        cls = cls[:, selected_indices]
        cls = cls.float().to(device=device)
        x_true=x_true.to(device=device)
        d_true = disc_model(x_true, cls)
        z = torch.randn(x_true.shape[0], LATENT_SIZE, device=device) 
        x_synth = gen_model(z, cls) 
        d_synth = disc_model(x_synth, cls)

        disc_optimizer.zero_grad() 
        dloss = disc_loss_function(d_true, d_synth)
        dloss.backward(retain_graph=True)
        disc_optimizer.step()  
        d_synth = disc_model(x_synth, cls) 
        gen_optimizer.zero_grad()
        gloss = gen_loss_function(d_synth) 
        gloss.backward()
        gen_optimizer.step()
        sum_gloss += gloss.detach().cpu().item()
        sum_dloss += dloss.detach().cpu().item()
        sum_dtrue += d_true.mean().detach().cpu().item() 
        sum_dsynth += d_synth.mean().detach().cpu().item() 
        batches += 1




if __name__ == '__main__':
    EPOCHS = 1000
    print(device)
    data_path = '.\\CELEBA-20250601T143311Z-1-001'
    save_path = ".\\models\\DDPM_models"

    transform=tforms.Compose([
	tforms.ToImage(),
    tforms.Resize((64,64)),
    tforms.ToDtype(torch.float32, scale=True), 
	tforms.Normalize(mean=[0.5]*3, std=[0.5]*3) 
	])
    training_set=CelebA(data_path, download=False, transform=transform)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
  
    save_interval = 30 * 60
    last_save_time = time.time()
    for i in range(EPOCHS):
        training_epoch(training_loader)
        print("EPOCA"+ str(i)+ " finita ")
        current_time = time.time()
        if current_time - last_save_time >= 30*60:
            torch.save(disc_model.state_dict(), save_path+'newtrydisc'+str(i)+'.pth')
            torch.save(gen_model.state_dict(), save_path+'newtrygen'+str(i)+'.pth')
            print(f'Salvataggio epoca ', i, ' completato.')
            last_save_time = current_time