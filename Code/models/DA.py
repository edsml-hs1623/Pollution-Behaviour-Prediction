from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import gc
from torch.utils.checkpoint import checkpoint


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


def sensitivity_analysis(mode, subdomain, cvae_model, ai4urban_model, optimizer, ntime, iter_sensitivity, window, latent_representations, obs_data, values_u, values_v, values_w, values_uu, values_vv, values_ww, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww):
    """
    Conducts sensitivity analysis on a conditional variational autoencoder (CVAE) model 
    by reconstructing the latent representations and performing simulations using the AI4Urban model.

    Args:
        model (str): The mode for 'velocity' or 'pollution'
        subdomain (str): The subdomain used, 'topleft', 'topright', 'bottomleft', 'bottomright'
        cvae_model (torch.nn.Module): The conditional variational autoencoder model used for reconstruction.
        ai4urban_model (torch.nn.Module): The AI4Urban model used for forward simulations based on the reconstructed data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the parameters of the CVAE model.
        ntime (int): The number of time steps for the AI4Urban model simulation.
        iter_sensitivity (int): The number of sensitivity analysis iterations to perform.
        window (str): The window name or identifier used for saving outputs.
        latent_representations (torch.Tensor): The latent representations of the data used for reconstruction.
        obs_data (np.ndarray): The observed data used for comparison against the model outputs.
        values_u (torch.Tensor): Initial values for the u-component of the velocity flow.
        values_v (torch.Tensor): Initial values for the v-component of the velocity flow.
        values_w (torch.Tensor): Initial values for the w-component of the velocity flow.
        values_uu (torch.Tensor): Initial values for the uu-component of the velocity flow.
        values_vv (torch.Tensor): Initial values for the vv-component of the velocity flow.
        values_ww (torch.Tensor): Initial values for the ww-component of the velocity flow.
        values_pp (torch.Tensor): Initial values for the pressure component of velocityfluid flow.
        b_uu (torch.Tensor): Boundary condition values for the uu-component.
        b_vv (torch.Tensor): Boundary condition values for the vv-component.
        b_ww (torch.Tensor): Boundary condition values for the ww-component.
        k1 (torch.Tensor): Coefficient values used in the simulation.
        dt (float): Time step for the simulation.
        iteration (int): Current iteration number.
        k_uu (torch.Tensor): Coefficient values for the uu-component.
        k_vv (torch.Tensor): Coefficient values for the vv-component.
        k_ww (torch.Tensor): Coefficient values for the ww-component.

    Returns:
        tuple: A tuple containing:
            - original_data (torch.Tensor): The original data reconstructed during the sensitivity analysis.
            - recon_progress (list): A list containing the reconstructed values at each sensitivity analysis iteration.

    Notes:
        - The function sets the gradients for parameters in the decoder part of the CVAE model to `True` to enable backpropagation.
        - Anomaly detection is enabled during the forward pass to catch any potential issues.
        - The function reconstructs data using the CVAE model and simulates the results using the AI4Urban model for the specified number of time steps.
        - It also computes relative errors between simulated values and observed data at specific sensor locations.
        - The output of the model is saved at the end of the sensitivity analysis iterations, along with the model state.

    Example:
        original_data, recon_progress = sensitivity_analysis(cvae_model, ai4urban_model, optimizer, ntime, iter_sensitivity,
                                                            window, latent_representations, obs_data, values_u, values_v,
                                                            values_w, values_uu, values_vv, values_ww, values_pp,
                                                            b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww,
                                                            save_path)
    """

    # the param from decoder is what we aimed for
    for name, param in cvae_model.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    recon_progress= []
    for iter in range(iter_sensitivity):
        optimizer.zero_grad()

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # reconstruct the subdomains 64x64 with decoder
        original_data, reconstructed_data = reconstruct(cvae_model, test_loader, latent_repr=latent_representations_tensor)
        print(f"\nreconstructed data shape: {reconstructed_data.shape}")

        # reconstruct the full 512x512 image as initial condition for CFD simulation
        reconstructed_dataset = TensorDataset(reconstructed_data)
        reconstructed_loader = DataLoader(reconstructed_dataset, batch_size=49)
        reconstructed_full_data, _ = reconstruct_full_data(reconstructed_loader, (1, 3, 64, 256, 256), block_size=64, overlap_ratio=(0.5, 0.5))
        print(f"reconstructed_full shape: {reconstructed_full_data.shape}")


        # Create clones to avoid in-place modifications, Detach unecessary tensors
        if mode == 'velocity':
                values_u_clone = reconstructed_data[0, 0, :, :, :].unsqueeze(0).unsqueeze(0).clone().to(device) ; values_v_clone = reconstructed_data[0, 1, :, :, :].unsqueeze(0).unsqueeze(0).clone().to(device) ; values_w_clone = reconstructed_data[0, 2, :, :, :].unsqueeze(0).unsqueeze(0).clone().to(device)
                values_p_clone = values_p.clone().detach().to(device) ; values_pp_clone = values_pp.clone().detach().to(device) ; k1_clone = k1.clone().detach().to(device)
        elif mode == 'pollution':
                values_u_clone = values_u.clone().detach().to(device) ; values_v_clone = values_v.clone().detach().to(device) ; values_w_clone = values_w.clone().detach().to(device)
                values_p_clone = reconstructed_full_data[0, 0, :, :, :].clone().unsqueeze(0).unsqueeze(0).to(device) ; values_pp_clone = values_pp.clone().detach().to(device) ; k1_clone = k1.clone().detach().to(device)
        
        values_uu_clone = values_uu.clone().detach().to(device) ; values_vv_clone = values_vv.clone().detach().to(device) ; values_ww_clone = values_ww.clone().detach().to(device)
        b_uu_clone = b_uu.clone().detach().to(device) ; b_vv_clone = b_vv.clone().detach().to(device) ; b_ww_clone = b_ww.clone().detach().to(device)
        k_uu_clone = k_uu.clone().detach().to(device) ; k_vv_clone = k_vv.clone().detach().to(device) ; k_ww_clone = k_ww.clone().detach().to(device)


        # Forward pass through AI4Urban model with generated initial condition
        for itime in range(1, ntime+1):

            # Place constant source of pollution before each iteration
            if mode == 'pollution':
                values_p = create_pollution_source(values_p, start_row=320, end_row=350, start_col=100, end_col=225, value=1.5, slope_row=-1, slope_col=5, thickness=6, device=device)
                values_p = create_pollution_source(values_p, start_row=290, end_row=350, start_col=220, end_col=400, value=0.5, slope_row=-1, slope_col=4, thickness=2, device=device)
                values_p = create_pollution_source(values_p, start_row=310, end_row=350, start_col=230, end_col=400, value=1.0, slope_row=-1, slope_col=4, thickness=3, device=device)
                values_p = create_pollution_source(values_p, start_row=180, end_row=310, start_col=212, end_col=380, value=0.5, slope_row=9, slope_col=1, thickness=2, device=device)
                values_p = create_pollution_source(values_p, start_row=185, end_row=315, start_col=212, end_col=380, value=0.5, slope_row=9, slope_col=1, thickness=2, device=device)

            # CFD simulation
            values_u, values_v, values_w, values_p, w, r = checkpointed_forward(ai4urban_model, values_u_clone, values_uu_clone, values_v_clone, values_vv_clone, values_w_clone, values_ww_clone, values_p_clone, values_pp_clone, b_uu_clone, b_vv_clone, b_ww_clone, k1_clone, dt, iteration, k_uu_clone, k_vv_clone, k_ww_clone)
            
            # Create clones to avoid in-place modifications
            if mode == 'velocity':
                values_u_clone = values_u.clone().to(device); values_v_clone = values_v.clone().to(device); values_w_clone = values_w.clone().to(device)
            elif mode == 'pollution':
                values_p_clone = values_p.clone().to(device)

        print(f"\nai4urban iteration {ntime} done\n")


        # Concatenate model outputs and transform them into tensors
        if mode == 'velocity':
            values_sim = normalization(torch.cat((values_u, values_v, values_w), dim=1)).to(device)
            values_obs = normalization(torch.tensor(obs_data, dtype=torch.float)).to(device)
        elif mode == 'pollution':
            values_sim = normalization(torch.cat((values_p, values_p, values_p), dim=1)).to(device)  # duplicate into 3 channels to fit the cvae model
            values_obs = normalization(torch.tensor(obs_data, dtype=torch.float)).to(device)


        # Loss Function
        loss = nn.MSELoss()(values_sim, values_obs)
        print(f"\n----------Sensitivity Analysis Iteration {iter+1} Done, loss: {loss.item()}----------\n")

        # Save the output
        recon_progress.append(values_sim)

        # Backward pass to compute gradients
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(cvae_model.parameters(), 1.0)  # Clip gradients to avoid gradient explosion

        # Access gradients of leaf nodes from the decoder
        for name, param in cvae_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f"Max gradient of {name}: {torch.max(param.grad)}")
                else:
                    print(f"Gradient for {name} is None")

        # Update model parameters
        optimizer.step()
        print(f"\nModel parameters updated!")

        # Save the data after all iterations havd been done
        if iter == iter_sensitivity-1:
            np.save(os.path.join(f'/Output/{mode}', f"{mode}_{subdomain_}w{window}"), arr=values_sim.cpu().detach().numpy())
            torch.save(cvae_model.state_dict(), os.path.join(f'/Output/{mode}', f"{mode}_{subdomain}_w{window}.pth"))

        # Free up cache after each iteration
        torch.cuda.empty_cache()
        gc.collect()

    return original_data, recon_progress
