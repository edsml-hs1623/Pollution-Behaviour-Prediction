import numpy as np
import sys
import os

from notebook.utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from livelossplot import PlotLosses
import cv2
import random

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


def create_zigzag_pattern(map, start_row, end_row, start_col, end_col, z_index, value, slope_row=1, slope_col=1, thickness=2, device='cpu'):
    row = start_row
    col = start_col

    while row < end_row and col < end_col:
        for i in range(-thickness, thickness + 1):
            for j in range(-thickness, thickness + 1):
                if (0 <= row + i < map.shape[3]) and (0 <= col + j < map.shape[4]):
                    map[0, 0, z_index, row + i, col + j] = torch.tensor(value, dtype=torch.float32, device=device)
        row += slope_row
        col += slope_col

    return map

def show_overlapping_p(data, start_index, z):
    # channel_titles = ['Channel 1', 'Channel 2', 'Channel 3']

    fig, axes = plt.subplots(4, 4)
    for row in range(4):
        for col in range(4):
            idx = start_index + row * 7 + col  # Calculate the correct index for each subplot
            ax = axes[row, col]
            ax.imshow(data[idx][0, z].squeeze())
            ax.axis('off')

            # Add title to the first column of each row
            # if col == 0:
            #     ax.set_title(channel_titles[row], loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_recon_progress(recon_progress, obs_data):
    random_indices = [3, 13, 23, 33, 43, 53, 63]
    num_channels = 3

    # Plot original and reconstructed images
    fig, axes = plt.subplots(len(recon_progress)+1, len(random_indices), figsize=(20, 20))

    # initial condition
    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(obs_data[0, 0, idx, :, :]) #.detach().numpy())
        axes[0, i].set_title(f'Init z = {idx + 1}')
        axes[0, i].axis('off')

    for iter in range(len(recon_progress)):
        for i, idx in enumerate(random_indices):
            reconstructed = recon_progress[iter][0, 0, idx, :, :].detach().numpy()

            # Plot reconstructed image
            axes[iter, i].imshow(reconstructed)
            axes[iter, i].set_title(f'Iter = {iter * 3}, z = {idx + 1}')
            axes[iter, i].axis('off')

    # last row is the observation data
    for i, idx in enumerate(random_indices):
        axes[-1, i].imshow(obs_data[0, 0, idx, :, :]) #.detach().numpy())
        axes[-1, i].set_title(f'OBS z = {idx + 1}')
        axes[-1, i].axis('off')

    plt.tight_layout()
    plt.show()

def rotate_images(dataset):
    rotation_angles = [1, 2, 3]  # 90, 180, 270 degrees
    angle_index = 0

    for i, img in enumerate(dataset):
        if i % 18 == 0:  # Rotate every 18th image
            angle = rotation_angles[angle_index % len(rotation_angles)]
            dataset[i] = np.rot90(img, k=angle, axes=(2, 3))  # Rotate in-place
            angle_index += 1
            # print(f"Rotated image {i} by {angle} degrees.")

    return dataset

def show_overlapping(data, indices):
    channel_titles = ['Channel 1', 'Channel 2', 'Channel 3']

    fig, axes = plt.subplots(3, 7)
    for row in range(3):
        for col, idx in enumerate(indices):  # sequential images
            ax = axes[row, col]
            ax.imshow(data[idx][row, 4].squeeze())
            ax.axis('off')

            # Add title to the first column of each row
            if col == 0:
                ax.set_title(channel_titles[row], loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()

def reconstruct_full_data(blocks_array, data_shape, block_size, overlap_ratio):
    # Unpack data shape
    B, C, Z, X, Y = data_shape

    # Calculate overlap in terms of indices
    overlap_x = int(block_size * overlap_ratio[0])
    overlap_y = int(block_size * overlap_ratio[1])

    # Calculate number of blocks along each dimension
    X_blocks = (X - overlap_x) // (block_size - overlap_x)
    Y_blocks = (Y - overlap_y) // (block_size - overlap_y)

    # Initialize reconstructed full data array
    reconstructed_full_data = torch.zeros(data_shape)

    # Initialize count array to keep track of number of contributions to each pixel
    count_array = torch.zeros(data_shape)

    # try concatenating batches
    block_index = 0
    for batch_idx, batch_data in enumerate(blocks_array):
        batch_data_tensor = batch_data[0]
        batch_size = batch_data_tensor.shape[0]
        # print(f"Batch number {batch_idx}, batch size {batch_size}")

        for x in range(X_blocks):
            for y in range(Y_blocks):
                    x = block_index // Y_blocks
                    y = block_index % Y_blocks

                    x_start = x * (block_size - overlap_x)
                    x_end = x_start + block_size
                    y_start = y * (block_size - overlap_y)
                    y_end = y_start + block_size

                    for channel in range(C):
                        reconstructed_full_data[0, channel, :, x_start:x_end, y_start:y_end] += \
                            batch_data_tensor[block_index % 49, channel, :, :, :]
                        count_array[0, channel, :, x_start:x_end, y_start:y_end] += 1

                    # print(f"Placing block {block_index} from batch {batch_idx} into position ({y}, {x})")
                    block_index += 1

                    if block_index % 49 == 0:
                        break

            if block_index % 49 == 0:
                break

    reconstructed_full_data /= count_array

    return reconstructed_full_data, count_array

def data_downsample_overlap(data, block_size, overlap_ratio):
    # Get the shape of the combined data
    I, C, Z, X, Y = data.shape

    # Calculate overlap in terms of indices
    overlap_x = int(block_size * overlap_ratio[0])
    overlap_y = int(block_size * overlap_ratio[1])

    # Define the number of blocks along each dimension
    X_blocks = (X - overlap_x) // (block_size - overlap_x)
    Y_blocks = (Y - overlap_y) // (block_size - overlap_y)

    blocks = []

    # Iterate over the combined data to extract blocks
    for i in range(I):
        for x in range(X_blocks):
            for y in range(Y_blocks):
                x_start = x * (block_size - overlap_x)
                x_end = x_start + block_size
                y_start = y * (block_size - overlap_y)
                y_end = y_start + block_size

                block = data[i, :, :, x_start:x_end, y_start:y_end]
                blocks.append(block)

    # Convert the list of blocks to a numpy array
    blocks_array = np.array(blocks)

    return blocks_array

def split_data(blocks_array, rotation=False):

  # split into train, validation and test datasets (0.8, 0.1, 0.1)
  train_size = int(0.8 * len(blocks_array))
  val_size = int(0.1 * len(blocks_array))
  test_size = len(blocks_array) - train_size - val_size

  # Perform the splits
  train_data = blocks_array[:train_size]
  val_data = blocks_array[train_size:train_size + val_size]
  test_data = blocks_array[train_size + val_size:]

  # rotate some images in train set
  if rotation:
      print(f"before rotation: \n {train_data[0,0,4,0,0:5]} ...")  # first element = 1656 for 512, just show the first five values
      train_data = rotate_images(train_data)
      print(f"after rotation: \n {train_data[0,0,4,0,0:5]} ...")

  print(f"Train shape: {train_data.shape},\nValidation shape: {val_data.shape}, \nTest shape: {test_data.shape}")

  train_dataset = TensorDataset(torch.from_numpy(train_data))
  train_loader = DataLoader(train_dataset, batch_size=49, shuffle=False)

  val_dataset = TensorDataset(torch.from_numpy(val_data))
  val_loader = DataLoader(val_dataset, batch_size=49, shuffle=False)

  test_dataset = TensorDataset(torch.from_numpy(test_data))
  test_loader = DataLoader(test_dataset, batch_size=49, shuffle=False)

  return train_loader, val_loader, test_loader

def visualize_data(data):
    indices = [1, 2, 3, 4, 20, 40, 60]
    channel_titles = ['Channel 1', 'Channel 2', 'Channel 3']

    fig, axes = plt.subplots(3, 7)
    for row in range(3):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            if isinstance(data, torch.Tensor):
                ax.imshow(data[row, idx].detach().numpy().squeeze())
            elif isinstance(data, np.ndarray):
                ax.imshow(data[row, idx].squeeze())
            else:
                raise ValueError("Unsupported data type")
            ax.axis('off')

            # Add title to the first column of each row
            if col == 0:
                ax.set_title(channel_titles[row], loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_p(data):
    indices = [1, 2, 3, 4, 20, 40, 60]

    fig, axes = plt.subplots(1, 7)
    for col, idx in enumerate(indices):
        ax = axes[col]
        if isinstance(data, torch.Tensor):
            ax.imshow(data[0, idx].detach().numpy().squeeze())
        elif isinstance(data, np.ndarray):
            ax.imshow(data[0, idx].squeeze())
        else:
            raise ValueError("Unsupported data type")
        ax.axis('off')

        # Add title to the first column of each row
        if col == 0:
            ax.set_title("pollution", loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()

def normalization(data):

    # Calculate global min and max values across all elements
    if isinstance(data, np.ndarray):
        global_min_val = np.min(data)
        global_max_val = np.max(data)
    elif isinstance(data, torch.Tensor):
        global_min_val = torch.min(data)
        global_max_val = torch.max(data)

    # Normalize the entire dataset to (0, 1) using global min and max
    data = (data - global_min_val) / (global_max_val - global_min_val)

    return data

def combine_and_save_data(base_path, prefixes, steps):
    for step in steps:
        # Load the data for each prefix at the current time step
        data_step = []
        for prefix in prefixes:
            file_path = os.path.join(base_path, f"{prefix}{step}.npy")
            data = np.load(file_path)
            data_step.append(data)

        # Stack the data arrays for the current time step along a new axis
        stacked_data = np.stack(data_step, axis=0)

        # Save the combined data to a new file
        save_filename = f"combined_data_{step}.npy"
        save_file_path = os.path.join(base_path, save_filename)
        np.save(save_file_path, stacked_data)

        print(f"Combined data for step {step} shape: {stacked_data.shape}")
        print(f"Combined data for step {step} saved to: {save_file_path}")

def final_combine(base_path, steps):
    normalized_data_list = []

    for step in steps:
        # Load the data
        file_path = os.path.join(base_path, f"combined_data_{step}.npy")
        data = np.load(file_path)

        # Normalize the data
        normalized_data = normalization(data)

        # Append normalized data to the list
        normalized_data_list.append(normalized_data)

    # Stack the normalized data arrays along the first axis
    data_streets = np.stack(normalized_data_list, axis=0)

    print(f"Final combined data shape: {data_streets.shape}")

    # Example of saving the final combined and normalized data
    np.save(os.path.join(base_path, "final_combined_data_p.npy"), data_streets)
    print("Final combined and normalized data saved.")

def data_downsample(data, block_size):
    # Get the shape of the combined data
    I, C, Z, X, Y = data.shape

    # Define the block size and the number of blocks along each dimension
    Z_blocks = Z
    X_blocks = X // block_size
    Y_blocks = Y // block_size

    blocks = []

    # Iterate over the combined data to extract blocks
    for i in range(I):
        for x in range(X_blocks):
            for y in range(Y_blocks):
                block = data[i, :, :, x*block_size:(x+1)*block_size, y*block_size:(y+1)*block_size]
                blocks.append(block)

    # Convert the list of blocks to a numpy array
    blocks_array = np.array(blocks)

    return blocks_array

def visualize_da_result(recon_progress, init_data, obs_data, channel):
    random_indices = [1, 2, 3, 4, 20]

    # Plot original and reconstructed images
    fig, axes = plt.subplots(3, len(random_indices))

    # initial condition
    for i, idx in enumerate(random_indices):
        if torch.is_tensor(init_data):
            axes[0, i].imshow(init_data[channel, idx, :, :].cpu().detach().numpy())
        else:
            axes[0, i].imshow(init_data[channel, idx, :, :])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Simulation", loc='left', fontsize=12)


    # recon progress
    for i, idx in enumerate(random_indices):
        if torch.is_tensor(recon_progress):
            reconstructed = recon_progress[channel, idx, :, :].cpu().detach().numpy()
        else:
            reconstructed = recon_progress[channel, idx, :, :]
        axes[-1, i].imshow(reconstructed)
        axes[-1, i].axis('off')
        if i == 0:
            axes[-1, i].set_title(f"Updated", loc='left', fontsize=12)

    # last row is the observation data
    for i, idx in enumerate(random_indices):
        if torch.is_tensor(obs_data):
            axes[1, i].imshow(obs_data[channel, idx, :, :].cpu().detach().numpy())
        else:
            axes[1, i].imshow(obs_data[channel, idx, :, :])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Observation", loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_da_detail(recon_progress, init_data, obs_data, channel, x_start, x_end, y_start, y_end):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # initial condition
    if torch.is_tensor(init_data):
        init_region = init_data[channel, 1, x_start:x_end, y_start:y_end].cpu().detach().numpy()
    else:
        init_region = init_data[channel, 1, x_start:x_end, y_start:y_end]
    axes[0].imshow(init_region)
    axes[0].axis('off')
    axes[0].set_title("Simulation", fontsize=12)

    # observation data
    if torch.is_tensor(obs_data):
        obs_region = obs_data[channel, 1, x_start:x_end, y_start:y_end].cpu().detach().numpy()
    else:
        obs_region = obs_data[channel, 1, x_start:x_end, y_start:y_end]
    axes[1].imshow(obs_region)
    axes[1].axis('off')
    axes[1].set_title("Observation", fontsize=12)

    # reconstruction progress
    if torch.is_tensor(recon_progress):
        recon_region = recon_progress[channel, 1, x_start:x_end, y_start:y_end].cpu().detach().numpy()
    else:
        recon_region = recon_progress[channel, 1, x_start:x_end, y_start:y_end]
    axes[2].imshow(recon_region)
    axes[2].axis('off')
    axes[2].set_title(f"Updated", fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_sensor(mismatch, s1c1, s1c2, s1c3, s1p, s2c1, s2c2, s2c3, s2p, s3c1, s3c2, s3c3, s3p, s4c1, s4c2, s4c3, s4p, s5c1, s5c2, s5c3, s5p):
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes[0,0].plot(mismatch, marker='o')
    axes[0,0].set_title('Data Mismatch', fontsize=18)
    axes[0,0].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[0,0].set_ylabel('Value', fontsize=12)

    axes[0,1].plot(s1c1, marker='o', label='channel 1')
    axes[0,1].plot(s1c2, marker='x', label='channel 2')
    axes[0,1].plot(s1c3, marker='s', label='channel 3')
    axes[0,1].plot(s1p, marker='*', label='pollution')
    axes[0,1].legend()
    axes[0,1].set_title('Sensor 1 (156, 313)', fontsize=18)
    axes[0,1].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[0,1].set_ylabel('Relative Error', fontsize=12)

    axes[1,0].plot(s2c1, marker='o', label='channel 1')
    axes[1,0].plot(s2c2, marker='x', label='channel 2')
    axes[1,0].plot(s2c3, marker='s', label='channel 3')
    axes[1,0].plot(s2p, marker='*', label='pollution')
    axes[1,0].legend()
    axes[1,0].set_title('Sensor 2 (276, 304)', fontsize=18)
    axes[1,0].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[1,0].set_ylabel('Relative Error', fontsize=12)


    axes[1,1].plot(s3c1, marker='o', label='channel 1')
    axes[1,1].plot(s3c2, marker='x', label='channel 2')
    axes[1,1].plot(s3c3, marker='s', label='channel 3')
    axes[1,1].plot(s3p, marker='*', label='pollution')
    axes[1,1].legend()
    axes[1,1].set_title('Sensor 3 (375, 289)', fontsize=18)
    axes[1,1].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[1,1].set_ylabel('Relative Error', fontsize=12)

    axes[2,0].plot(s4c1, marker='o', label='channel 1')
    axes[2,0].plot(s4c2, marker='x', label='channel 2')
    axes[2,0].plot(s4c3, marker='s', label='channel 3')
    axes[2,0].plot(s4p, marker='*', label='pollution')
    axes[2,0].legend()
    axes[2,0].set_title('Node 1 (255, 150)', fontsize=18)
    axes[2,0].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[2,0].set_ylabel('Relative Error', fontsize=12)

    axes[2,1].plot(s5c1, marker='o', label='channel 1')
    axes[2,1].plot(s5c2, marker='x', label='channel 2')
    axes[2,1].plot(s5c3, marker='s', label='channel 3')
    axes[2,1].plot(s5p, marker='*', label='pollution')
    axes[2,1].legend()
    axes[2,1].set_title('Node 2 (255, 400)', fontsize=18)
    axes[2,1].set_xlabel('Data Assimilation Iteration', fontsize=12)
    axes[2,1].set_ylabel('Relative Error', fontsize=12)
    plt.show()