import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from itertools import product
from CVAE3brb import AutoEncoder

LATENT_SIZE = 512
device = "cuda"
model_name = "./VAE/vae_brb252.pth"




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(LATENT_SIZE).to(device=device)
checkpoint = torch.load(model_name, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conditions = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
], device=device)

#conditions = torch.tensor([[0,0,1]]*8, device=device)

num_conditions = conditions.shape[0]
z = torch.randn(num_conditions, LATENT_SIZE, device=device).to(device)
c = conditions  

with torch.no_grad():
    z_cond = torch.cat([z, c], dim=1) 
    z_cond = z_cond.to(device)
    z2=model.decoder_input(z_cond)
    z_conc = model.concatenate_cond(x=z2, cond=c)
    imgs=model.decoder(z_conc)
    imgs = imgs.cpu()

imgs = imgs.permute(0, 2, 3, 1) 

n = len(conditions)
cols = min(4, n)
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols * 3, rows * 3))

for i in range(n):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(imgs[i].numpy())
    plt.axis('off')
    plt.title(f"cond: {conditions[i].tolist()}")

plt.tight_layout()
plt.show()