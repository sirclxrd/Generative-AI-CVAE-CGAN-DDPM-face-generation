import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as tforms
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

import math

device='cuda'
IMAGE_SHAPE=(3,64,64)
IMAGE_DIMENSIONS=(1,1,1)
COND_SHAPE=(3,)




class NoiseSchedule:
    def __init__(self, L, s=0.008, device=device):
        self.L=L
        t=torch.linspace(0.0, L, L+1, device=device)/L
        a=torch.cos((t+s)/(1+s)*torch.pi/2)**2
        a=a/a[0]
        self.beta=(1-a[1:]/a[:-1]).clip(0.0, 0.99)
        self.alpha=torch.cumprod(1.0-self.beta, dim=0)
        self.one_minus_beta=1-self.beta
        self.one_minus_alpha=1-self.alpha
        self.sqrt_alpha=torch.sqrt(self.alpha)
        self.sqrt_beta=torch.sqrt(self.beta)
        self.sqrt_1_alpha=torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta=torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L

# Number of steps
L=1000
noise_schedule=NoiseSchedule(L)

print('sqrt(alpha_L)=', noise_schedule.sqrt_alpha[-1].cpu().item())

TIME_ENCODING_SIZE=64

class TimeEncoding:
    def __init__(self, L, dim, device=device):
        # Note: the dimension dim should be an even number
        self.L=L
        self.dim=dim
        dim2=dim//2
        encoding=torch.zeros(L, dim)
        ang=torch.linspace(0.0, torch.pi/2, L)
        logmul=torch.linspace(0.0, math.log(40), dim2)
        mul=torch.exp(logmul)
        for i in range(dim2):
            a=ang*mul[i]
            encoding[:,2*i]=torch.sin(a)
            encoding[:,2*i+1]=torch.cos(a)
        self.encoding=encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]


time_encoding=TimeEncoding(L, TIME_ENCODING_SIZE)


class UNetBlock(nn.Module):
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        self.encoder = self.build_encoder(outer_features+cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features+cond_features+TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2*outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, cond):
        x0=x
        male, glasses, no_beard = cond[:, 0], cond[:, 1], cond[:, 2]
        male_tensor = male.view(-1, 1, 1, 1).expand(-1, 1, self.size, self.size).to(x.device)
        glasses_tensor = glasses.view(-1, 1, 1, 1).expand(-1, 1, self.size, self.size).to(x.device)
        no_beard_tensor = no_beard.view(-1, 1, 1, 1).expand(-1, 1, self.size, self.size).to(x.device)
        cc = torch.cat([male_tensor, glasses_tensor, no_beard_tensor], dim=1)
        x=torch.cat((x,cc), dim=1)
        y=self.encoder(x)
        if self.inner:
            y=self.inner(y, time_encodings, cond)
        half_size=self.size//2
        male_tensor = male.view(-1, 1, 1, 1).expand(-1, 1, half_size,half_size).to(x.device)
        glasses_tensor = glasses.view(-1, 1, 1, 1).expand(-1, 1, half_size, half_size).to(x.device)
        no_beard_tensor = no_beard.view(-1, 1, 1, 1).expand(-1, 1, half_size,half_size).to(x.device)
        tt=time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        cc = torch.cat([male_tensor, glasses_tensor, no_beard_tensor], dim=1)
        y1=torch.cat((y,cc,tt), dim=1)
        x1=self.decoder(y1)
        x2=torch.cat((x1,x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        model=nn.Sequential(
                nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                nn.BatchNorm2d(from_features),
                nn.ReLU(),
                nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(to_features),
                nn.ReLU()
        )
        return model

    def build_decoder(self, from_features, to_features):
        model=nn.Sequential(
                nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                nn.BatchNorm2d(from_features),
                nn.ReLU(),
                nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(to_features),
                nn.ReLU()
        )
        return model




class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),
            nn.ReLU())
        self.unet = self.build_unet(64, [64, 128, 256, 512, 1024]) #size, feat_list, ********
        self.post=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding='same'))

    def forward(self, x, t, cond):
        enc=time_encoding[t]
        x=self.pre(x)
        y=self.unet(x, enc, cond)
        y=self.post(y)
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list)>2:
            inner_block=self.build_unet(size//2, feat_list[1:])
        else:
            inner_block=None
        return UNetBlock(size, feat_list[0], feat_list[1], COND_SHAPE[0], inner_block)





model=Network()
model=model.to(device=device)

loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

epoch_count=0
selected_indices = [15, 20, 24]
def training_epoch(dataloader):
    global epoch_count
    model.train()
    average_loss=0.0
    batch = 0
    for x, cls in dataloader:
        x=x.to(device=device)
        n=x.shape[0] # Minibatch size
        cls = cls[:, selected_indices]
        cls = cls.float().to(device=device)
        

        # Remove the conditioning information with probability P=0.2
        P=0.2
        u=torch.rand((n,))
        cls[u<P,:]=2

        # Generate the random step indices (one for each sample in the minibatch)
        t=torch.randint(0, L, (n,), device=device)

        # Generate the random noise
        eps=torch.randn_like(x)

        # Compute latent image
        sqrt_alpha=noise_schedule.sqrt_alpha[t].view(-1, *IMAGE_DIMENSIONS)
        sqrt_1_alpha=noise_schedule.sqrt_1_alpha[t].view(-1, *IMAGE_DIMENSIONS)
        zt=sqrt_alpha*x + sqrt_1_alpha*eps

        # dopo aver creato l'immagine rumorosa, stima il rumore su ogni pixel 
        g=model(zt,t,cls)
        loss=loss_function(g, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_loss=0.9*average_loss+0.1*loss.cpu().item()
        print("batch: ",batch)
        batch += 1
    epoch_count += 1
    with open("DDPMCONDnew.txt", "a") as f:
        f.write(f'Epoch {epoch_count} completed. Average loss={average_loss}')

if __name__ == '__main__':
    EPOCHS = 100
    print(device)
    data_path = "/home/pfoggia/GenerativeAI/CELEBA/"
    data_path = './Data/CelebA'
    #save_path = "/home/C.DEANGELIS29/cond_test/DDPM_models/"
    save_path = "/home/G.CASELLA10/cond_test/VAE_models/"
    
    save_path = './DDPM/'
    last_epoch = 0


    transform = tforms.Compose([
    tforms.ToImage(),
    tforms.Resize((64, 64)),
    tforms.ToDtype(torch.float32, scale=True),
    ])
    training_set=CelebA(data_path, download=False, transform=transform)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)

    # checkpoint = torch.load(load_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # last_epoch = checkpoint['epoch'] + 1  # Per riprendere
    

    save_interval = 20 * 60
    last_save_time = time.time()
    for i in range(EPOCHS):
        training_epoch(training_loader)
        print("EPOCA"+ str(i)+ " finita ")
        current_time = time.time()
        if current_time - last_save_time >= 20*60 or i == 99:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_path + "ddpnew" + str(i + last_epoch) + ".pth")
            print(f'Salvataggio epoca ', i, ' completato.')
            last_save_time = current_time