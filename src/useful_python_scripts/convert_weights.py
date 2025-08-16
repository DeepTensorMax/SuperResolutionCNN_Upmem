import torch

checkpoint = torch.load('/workspaces/upmem-image-rescaling/ressources/srcnn_x2.pth', map_location=torch.device('cpu'))

def save_to_binary(filename, data):
    with open(filename, 'wb') as f:
        data.tofile(f)

for i in range(1, 4):
    save_to_binary(f'/workspaces/upmem-image-rescaling/ressources/bin/layer{i}_weights_x2.bin', checkpoint[f'conv{i}.weight'].cpu().numpy())
    save_to_binary(f'/workspaces/upmem-image-rescaling/ressources/bin/layer{i}_biases_x2.bin', checkpoint[f'conv{i}.bias'].cpu().numpy())
