import CFD
import CVAE
from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


#################### Parameters for Data Assimilation ####################

### Parameters that have to edit manually
mode = 'velocity'      # 'velocity', 'pollution'
subdomain = 'topleft'  # 'topleft', 'topright', 'bottomleft', 'bottomright'
window = 1             # index of the moving window

### Fixed parameters
iter_sensitivity = 5   # sensitivity analysis
ntime = 3000           # ai4urban time steps


#################### Load in  Models ####################
# AI4Urban model
AI4Urban_model = CFD.AI4Urban().to(device)
optimizer = optim.Adam(AI4Urban_model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization with weight decay
criterion = nn.MSELoss()

# Convolutional VAE
cvae_model = CVAE.Conv3DAutoencoder().to(device)
cvae_model.load_state_dict(torch.load(f'../Du Cane Road/{mode}_{subdomain}.pth', map_location=device))


#################### Initialise Input Tensors ####################
dt = 0.5
iteration = 5  # for multigird method
input_shape = (1, 1, 64, 128, 128)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
values_w = torch.zeros(input_shape, device=device)
values_p = torch.zeros(input_shape, device=device)
w = torch.zeros(input_shape, device=device)
r = torch.zeros(input_shape, device=device)
k1 = torch.ones(input_shape, device=device) * 2.0
input_shape_pad = (1, 1, 64+2, 128+2, 128+2)
values_uu = torch.zeros(input_shape_pad, device=device)
values_vv = torch.zeros(input_shape_pad, device=device)
values_ww = torch.zeros(input_shape_pad, device=device)
values_pp = torch.zeros(input_shape_pad, device=device)
b_uu = torch.zeros(input_shape_pad, device=device)
b_vv = torch.zeros(input_shape_pad, device=device)
b_ww = torch.zeros(input_shape_pad, device=device)
k_uu = torch.zeros(input_shape_pad, device=device)
k_vv = torch.zeros(input_shape_pad, device=device)
k_ww = torch.zeros(input_shape_pad, device=device)


#################### Load in Data ####################
# Whole Dataset (10, 3, 64, 512, 512)
if mode == 'velocity':
    dataset = np.load("../Du Cane Road/final_combined_data.npy")
elif mode == 'pollution':
    dataset = np.load("../Du Cane Road/final_combined_data_p.npy")

if subdomain == 'topleft':
    dataset = dataset[:, :, :, :256, :256]
elif subdomain == 'topright':
    dataset = dataset[:, :, :, :256, 256:]
elif subdomain == 'bottomleft':
    dataset = dataset[:, :, :, 256:, :256]
elif subdomain == 'bottomright':
    dataset = dataset[:, :, :, 256:, 256:]

# Observational Data
obs_data = dataset[9]  # (3, 64, 256, 256)
obs_data = np.expand_dims(obs_data, axis=0)  # (1, 3, 64, 256, 256)

# Test Data (With moving window strategy to overcome the memory issue)
if window == 1:
    test_data = dataset[0]  # (3, 64, 256, 256)
    test_data_original = np.expand_dims(test_data, axis=0)  # (1, 3, 64, 256, 256)
else:
    test_data_original = np.load(f"../Output/{mode}/{mode}_{subdomain}_w{window-1}.npy")  # continue previous iteration

test_data = data_downsample_overlap(test_data_original, block_size=64, overlap_ratio=(0.5, 0.5))  # (49, 3, 64, 64, 64)
test_dataset = TensorDataset(torch.from_numpy(test_data))
test_loader = DataLoader(test_dataset, batch_size=49, shuffle=False)

# Get Latent Representation from the CVAE model
latent_representations = []
for batch in test_loader:
    input_tensor = batch[0].to(device)
    latent_representation = get_latent_representation(cvae_model, input_tensor)
    latent_representations.append(latent_representation)
latent_representations_tensor = torch.stack(latent_representations)
print(f"The shape of latent representation: {latent_representations_tensor.shape}")


#################### Main Code ####################
# Run the Combined System that involves CVAE, CFD and DA
original_data, recon_progress = sensitivity_analysis(mode, subdomain, cvae_model, AI4Urban_model, optimizer, 
                                                     ntime, iter_sensitivity, window, latent_representations, 
                                                     obs_data, values_u, values_v, values_w, values_uu, values_vv, 
                                                     values_ww, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww)

# Save the Result
# visualize_recon_progress(recon_progress, test_data_original, obs_data, iter_sensitivity, ntime, window)
