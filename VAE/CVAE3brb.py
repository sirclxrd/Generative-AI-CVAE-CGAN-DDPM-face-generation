import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as tforms
from torchvision.datasets import CelebA
from matplotlib import pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not device:
    device='cpu'
print(device)
EYEGLASSES_FEATURE=15
GENDER_FEATURE=20
NO_BEARD_FEATURE=24

LATENT_SIZE=512
COND_SIZE = 5
COND_SIZE_DECODER = 3

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size=latent_size
        self.cond_size = COND_SIZE
        self.encoder=self.build_encoder(latent_size)
        self.decoder_input, self.decoder=self.build_decoder(latent_size)

    


    def concatenate_cond(self, x, cond, H=None, W=None):
        if H is None or W is None:
            _, _, H, W = x.shape  

        device = x.device
        cond1 = cond[:, 0].view(-1, 1, 1, 1).expand(-1, 1, H, W).to(device)  # male
        cond2 = cond[:, 1].view(-1, 1, 1, 1).expand(-1, 1, H, W).to(device)  # glasses
        cond3 = cond[:, 2].view(-1, 1, 1, 1).expand(-1, 1, H, W).to(device)  # no_beard
        mask = (cond[:, 2] == 0).float().view(-1, 1, 1, 1).to(device).expand(-1, 1, H, W)

        cond3_extra1 = mask
        cond3_extra2 = mask
        x_cond = torch.cat([x, cond1, cond2, cond3, cond3_extra1, cond3_extra2], dim=1)
        return x_cond




    def forward(self, x, cond):

        x_cond= self.concatenate_cond(x=x, cond=cond)
        out=self.encoder(x_cond)
        mu=out[:, :self.latent_size] 
        log_sigma=out[:, self.latent_size:]
        eps=torch.randn_like(mu)
        z=eps*torch.exp(log_sigma)+mu
        z_cond = torch.cat([z, cond], dim=1) 
        z2=self.decoder_input(z_cond)
        z_conc = self.concatenate_cond(x=z2, cond=cond)
        y=self.decoder(z_conc)
        return y, mu, log_sigma


    def build_encoder(self, latent_size):
        model=nn.Sequential() 
        prev=3 + COND_SIZE 
        size=64 
        k = prev
        for k in [64, 128, 256, 512, 1024]:
            model.append(nn.Conv2d(prev, k, 3, stride=2, padding=1))
            model.append(ResidualBlock(k))
            model.append(nn.BatchNorm2d(k))
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Dropout2d(p=0.2))
            prev=k
            size=size//2
        model.append(nn.Flatten())
        features=k*size*size
        model.append(nn.Linear(features, 2*latent_size))
        self.size_decoder = size
        return model

    # DECODER
    def build_decoder(self, latent_size):
        model_input=nn.Sequential()
        model_decoder=nn.Sequential()
        size=self.size_decoder
        prev=512
        latent_size = latent_size + COND_SIZE_DECODER 
        model_input.append(nn.Linear(latent_size, prev * size * size))
        model_input.append(nn.ReLU())
        model_input.append(nn.Unflatten(1, (prev, size, size)))
        prev = prev + COND_SIZE # z + cond
        for k in [1024,512,256,128, 64]:
            model_decoder.append(nn.ConvTranspose2d(prev, k, 3, stride=2, padding=1,
                                            output_padding=1))
            model_decoder.append(ResidualBlock(k))
            model_decoder.append(nn.BatchNorm2d(k))
            model_decoder.append(nn.LeakyReLU(0.2))
            model_decoder.append(nn.Dropout2d(p=0.2))
            prev=k
            size=size*2
        assert size==64 
        assert k==64
        model_decoder.append(nn.Conv2d(k, 3, 3, padding='same'))
        model_decoder.append(nn.Sigmoid())
        return model_input, model_decoder

# Model, optimizer and Loss Function
model=AutoEncoder(LATENT_SIZE).to(device=device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
rec_loss_function=nn.BCELoss(reduction='sum')

def kl_loss_function(mu, log_sigma):
    log_sigma2=2*log_sigma
    kl=mu**2 + torch.exp(log_sigma2) - 1.0 - log_sigma2
    return torch.sum(kl)

def loss_function(rec, target, mu, log_sigma):
    beta=1.0
    return rec_loss_function(rec, target) + beta*kl_loss_function(mu, log_sigma)

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch'] + 1  # Per riprendere

selected_indices = [15, 20, 24]
def training_epoch(data_loader, n_epochs = 0):

    num_total_batches = len(data_loader)
    average_loss=0.0
    batch_number = 0

    model.train()
    for x,lablel in data_loader:
        x=x.to(device=device)
        cond = lablel[:, selected_indices]
        cond = cond.float().to(device)
        optimizer.zero_grad()
        y, mu, log_sigma=model(x, cond)
        loss=loss_function(y, x, mu, log_sigma)
        loss.backward()
        optimizer.step()
        average_loss = 0.9*average_loss + 0.1*loss.cpu().item()
        batch_number += 1
        print('Epoch number', n_epochs, 'Batch number:', batch_number, '/', num_total_batches, 'Loss:', loss.item())
    scheduler.step()
    # with open("VAE_brb.txt", "a") as f:
    #     f.write(f'Epoch {n_epochs} completed. Average loss={average_loss}')
    n_epochs += 1
    return average_loss

if __name__ == '__main__':
    EPOCHS = 100
    print(device)
    #data_path = "/home/pfoggia/GenerativeAI/CELEBA/"
    data_path = "./DATA/celebA"
    #save_path = "/home/C.DEANGELIS29/cond_test/VAE_models/"
    save_path = "./CVAE_models/"
    #load_path = "/home/C.DEANGELIS29/cond_test/VAE_models/ddpm3Cond98.pth"
    last_epoch = 0
    loss = 1000


    transform = tforms.Compose([
    tforms.ToImage(),
    tforms.Resize((64, 64)),
    tforms.ToDtype(torch.float32, scale=True)    
    ])
    training_set=CelebA(data_path, download=False, transform=transform)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    save_interval = 50 * 60
    last_save_time = time.time()
    for i in range(EPOCHS):
        loss = training_epoch(training_loader, i)
        print("EPOCA"+ str(i)+ " finita ")
        current_time = time.time()
        if current_time - last_save_time >= 50*60 or i == (EPOCHS-1):
            torch.save({
                'loss': loss,
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_path + "vae_brb" + str(i + last_epoch) + ".pth")
            print(f'Salvataggio epoca ', i, ' completato.')
            last_save_time = current_time