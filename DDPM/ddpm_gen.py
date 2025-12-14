
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch



# Parametri (modifica se diversi)
IMAGE_SHAPE=(3,64,64)
IMAGE_DIMENSIONS=(1,1,1)
L = 1000



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "./DDPM/ddp280.pth"

from DDPMNEW import Network, TimeEncoding, NoiseSchedule
model = Network().to(device=device)
checkpoint = torch.load(model_name, map_location=device, weights_only=False) # carica dizionario con parametri del modello
model.load_state_dict(checkpoint['model_state_dict']) # specifica che ci interessano solo i pesi

cond_value = 2


model.eval()
time_encoding=TimeEncoding(L, 64)
noise_schedule=NoiseSchedule(L)



def generate(cond, lam=4.7):
    n=cond.shape[0]
    z=torch.randn(n, *IMAGE_SHAPE, device=device)
    cond0=torch.zeros_like(cond) + cond_value # diventa vettore di 2
    print(cond0)
    model.eval()
    for kt in reversed(range(L)):
        t=torch.tensor(kt).view(1).expand(n)

        beta=noise_schedule.beta[t].view(-1, *IMAGE_DIMENSIONS)
        sqrt_1_alpha=noise_schedule.sqrt_1_alpha[t].view(-1, *IMAGE_DIMENSIONS)
        sqrt_1_beta=noise_schedule.sqrt_1_beta[t].view(-1, *IMAGE_DIMENSIONS)
        sqrt_beta=noise_schedule.sqrt_beta[t].view(-1, *IMAGE_DIMENSIONS)

        g1 = model(z, t, cond)
        g0 = model(z, t, cond0)
        g = lam*g1 + (1-lam)*g0

        mu=(z-beta/sqrt_1_alpha*g)/sqrt_1_beta

        if kt>0:
            eps=torch.randn_like(z)
            z=mu+sqrt_beta*eps
            print(kt)
        else:
            z=mu
    return z




conditions = torch.tensor([
    [0, 0, 0],  # senza occhiali, donna, con barba
    [0, 0, 1],  # senza occhiali, donna, senza barba
    [0, 1, 0],  # senza occhiali, uomo, con barba
    [0, 1, 1],  # senza occhiali, uomo, senza barba
    [1, 0, 0],  # con occhiali, donna, con barba
    [1, 0, 1],  # con occhiali, donna, senza barba
    [1, 1, 0],  # con occhiali, uomo, con barba
    [1, 1, 1],  # con occhiali, uomo, senza barba
], dtype=torch.float32, device=device)

#conditions = torch.tensor([[0,0,1]]*8], device=device)

num_conditions = conditions.shape[0]

with torch.no_grad():
    gen = generate(conditions, 4.7)
    gen = gen.permute(0, 2, 3, 1)  # per riportare ordine canali da un tensore CxHxW
    imgs = gen.cpu()


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


