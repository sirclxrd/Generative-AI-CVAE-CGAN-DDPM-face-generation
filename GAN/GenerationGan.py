import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as tforms
LATENT_SIZE=256
COND_SHAPE=(3,)
from CGAN_newtry_c import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(LATENT_SIZE).to(device=device)
model.load_state_dict(torch.load("./GAN/modelgen68.pth", map_location=device, weights_only=False))
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

#conditions = torch.tensor([[0,0,1]]*8], device=device)

num_conditions = conditions.shape[0]
z = torch.randn(num_conditions, LATENT_SIZE, device=device).to(device)
c = conditions  

with torch.no_grad():
    imgs = model(z, c).cpu()  

imgs = imgs * 0.5 + 0.5  
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
