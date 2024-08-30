import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


def data_downsample_overlap(data, block_size, overlap_ratio):
    """
    Downsamples a multidimensional dataset into overlapping blocks.

    Args:
        data (numpy.ndarray): A 5D numpy array with shape (I, C, Z, X, Y), where:
                             - I is the number of images or samples,
                             - C is the number of channels,
                             - Z is the depth (number of slices or frames),
                             - X is the width of the data along one axis,
                             - Y is the height of the data along the other axis.
        block_size (int): The size of each block (assumed to be square) to extract from the data.
        overlap_ratio (tuple): A tuple of two floats representing the overlap ratios for the x and y dimensions.
                               For example, (0.5, 0.5) indicates 50% overlap.

    Returns:
        numpy.ndarray: A 5D numpy array containing the extracted blocks, with shape (N, C, Z, block_size, block_size),
                       where N is the total number of blocks extracted from the input data.

    Notes:
        - The function calculates the number of blocks to extract based on the specified overlap ratios,
          which determine how much the blocks overlap in the X and Y dimensions.
        - The resulting blocks are appended to a list and converted to a numpy array before being returned.
        - Ensure that the input data is appropriately shaped and that the specified block_size and overlap_ratio
          do not exceed the dimensions of the input data.
    """
    
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

def rotate_images(dataset):
    """
    Rotates every 18th image in the dataset by a predefined set of angles (90, 180, or 270 degrees).

    Args:
        dataset (numpy.ndarray): A 5D numpy array representing a batch of images with dimensions 
                                 (batch_size, channel, height, x, y). The images are assumed to be 
                                 in the format (N, C, H, X, Y), where N is the number of images, 
                                 C is the number of channels, H is the height, X is the width along 
                                 one axis, and Y is the width along the other axis.

    Returns:
        numpy.ndarray: The modified dataset with every 18th image rotated by 90, 180, or 270 degrees.
                       The rotation is performed in place, so the input dataset is altered.

    Notes:
        - The rotation is performed using `np.rot90`, which rotates the image counterclockwise.
        - The rotation angles alternate between 90, 180, and 270 degrees for every 18th image.
        - The rotation is applied along the last two dimensions (X, Y) of the images.
    """
    rotation_angles = [1, 2, 3]  # 90, 180, 270 degrees
    angle_index = 0

    for i, img in enumerate(dataset):
        if i % 18 == 0:  # Rotate every 18th image
            angle = rotation_angles[angle_index % len(rotation_angles)]
            dataset[i] = np.rot90(img, k=angle, axes=(2, 3))  # Rotate in-place
            angle_index += 1

    return dataset

def split_data(blocks_array, batch_size, rotation=True):
    """
    Splits a 5D array into training, validation, and test datasets and optionally rotates some images in the training set.

    Args:
        blocks_array (numpy.ndarray): A 5D numpy array with dimensions (N, C, H, X, Y), where:
                                      - N is the number of images,
                                      - C is the number of channels,
                                      - H is the height of the images,
                                      - X is the width of the images along one axis,
                                      - Y is the width of the images along the other axis.
        batch_size (int): The batch size to be used for the DataLoaders.
        rotation (bool, optional): If True, applies a rotation to some images in the training set using 
                                   the `rotate_images` function. The default is True.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.

    Notes:
        - The data is split into 80% training, 10% validation, and 10% test datasets.
        - If `rotation` is True, the `rotate_images` function is applied to rotate every 18th image in the training set.
        - DataLoaders are created with a batch size o 49 fand no shuffling.
    """

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
        train_data = rotate_images(train_data)

    print(f"Train shape: {train_data.shape},\nValidation shape: {val_data.shape}, \nTest shape: {test_data.shape}")

    train_dataset = TensorDataset(torch.from_numpy(train_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(torch.from_numpy(val_data))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.from_numpy(test_data))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_latent_representation(model, input_tensor):
    """
    Obtains the latent representation of an input tensor by passing it through the encoder of a trained model.

    Args:
        model (torch.nn.Module): The trained model containing an encoder component used to compute the latent representation.
        input_tensor (torch.Tensor): A tensor containing the input data. The shape of the tensor should be compatible with the model's encoder.

    Returns:
        torch.Tensor: The latent representation obtained from the encoder. This tensor's shape will depend on the architecture of the model's encoder.

    Notes:
        - The function disables gradient computation using `torch.no_grad()` to improve memory efficiency and speed during inference.
        - It sets the model to evaluation mode with `model.eval()`, ensuring that layers like dropout and batch normalization behave correctly.
        - The input tensor should be properly preprocessed and on the same device (CPU or GPU) as the model for correct execution.
    """
    with torch.no_grad():  # Disable gradient computation
        model.eval()  # Set the model to evaluation mode
        latent_representation = model.encoder(input_tensor)
    return latent_representation

def reconstruct(model, data_loader, latent_repr):
    """
    Reconstructs images from a dataset using a trained model by directly decoding a given latent representation.

    Args:
        model (torch.nn.Module): The trained model containing the decoder component.
        data_loader (DataLoader): A DataLoader providing the dataset for which the corresponding latent representations 
                                  are to be reconstructed.
        latent_repr (torch.Tensor): A tensor containing precomputed latent representations. The function assumes that 
                                    these latent representations correspond to the batches provided by the data_loader.

    Returns:
        tuple: A tuple containing:
            - original_data (torch.Tensor): A tensor containing the original images from the dataset, normalized.
            - reconstructed_data (torch.Tensor): A tensor containing the reconstructed images from the model, normalized.

    Notes:
        - The function assumes that `latent_repr` is not None and directly uses the provided latent representations 
          with the model's decoder to reconstruct the images.
        - The function processes each batch individually, accumulating the original and reconstructed images.
        - CUDA memory is manually cleared after processing each batch to manage GPU memory effectively.
    """

    model.eval()
    original_data = []
    reconstructed_data = []
    
    for idx, data in enumerate(data_loader):
        img = data[0].to(device)
        reconstructed = model.decoder(latent_repr[idx])

        original_data.append(img.cpu())
        reconstructed_data.append(reconstructed.cpu())

        del img, reconstructed
        torch.cuda.empty_cache()

    # Concatenate all batches into a single tensor for original and reconstructed data
    original_data = normalization(torch.cat(original_data))
    reconstructed_data = normalization(torch.cat(reconstructed_data))

    return original_data, reconstructed_data

def reconstruct_full_data(blocks_array, data_shape, block_size, overlap_ratio):
    """
    Reconstructs a full data volume from overlapping blocks using averaging to blend overlapping regions.

    Args:
        blocks_array (list or tensor): A collection of blocks (usually from a DataLoader) representing portions 
                                       of the original data volume.
        data_shape (tuple): The shape of the original data volume in the form (B, C, Z, X, Y), where:
                            - B is the batch size (usually 1 for a single volume),
                            - C is the number of channels,
                            - Z is the depth (or number of slices along the Z-axis),
                            - X is the width,
                            - Y is the height.
        block_size (int): The size of each block along both the X and Y dimensions.
        overlap_ratio (tuple): The overlap ratio for blocks along the X and Y dimensions, expressed as a fraction 
                               of the block size (e.g., (0.5, 0.5) for 50% overlap).

    Returns:
        tuple: A tuple containing:
            - reconstructed_full_data (torch.Tensor): The reconstructed full data volume, with overlapping regions 
                                                      averaged.
            - count_array (torch.Tensor): An array tracking the number of contributions to each pixel, used for averaging.

    Notes:
        - The function assumes that the blocks are arranged in a regular grid with specified overlap along the X and Y dimensions.
        - The reconstruction is performed by summing the overlapping blocks and dividing by the number of contributions 
          to each pixel to average out the overlaps.
        - The `count_array` is used to normalize the overlapping regions by keeping track of how many blocks contribute 
          to each pixel.
        - The reconstruction process considers the overlap and properly aligns the blocks to their original positions 
          in the full data volume.
    """

    # Unpack data shape
    B, C, Z, X, Y = data_shape

    # Calculate overlap in terms of indices
    overlap_x = int(block_size * overlap_ratio[0])
    overlap_y = int(block_size * overlap_ratio[1])

    # Calculate number of blocks along each dimension
    X_blocks = (X - overlap_x) // (block_size - overlap_x)
    Y_blocks = (Y - overlap_y) // (block_size - overlap_y)

    # Initialize reconstructed full data array
    reconstructed_full_data = torch.zeros(data_shape, dtype=torch.float)

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
                            batch_data_tensor[block_index % batch_size, channel, :, :, :]
                        count_array[0, channel, :, x_start:x_end, y_start:y_end] += 1

                    # print(f"Placing block {block_index} from batch {batch_idx} into position ({y}, {x})")
                    block_index += 1

                    if block_index % batch_size == 0:
                        break

            if block_index % batch_size == 0:
                break

    reconstructed_full_data /= count_array

    return reconstructed_full_data, count_array

def normalization(data):
    """
    Normalizes the input data to a range of [0, 1] using the global minimum and maximum values.

    Args:
        data (numpy.ndarray or torch.Tensor): The input data to be normalized. The data can be a 
                                              numpy array or a PyTorch tensor.

    Returns:
        numpy.ndarray or torch.Tensor: The normalized data, with the same type as the input.

    Notes:
        - The normalization is performed by scaling the data based on its global minimum and maximum values, 
          using the formula: normalized_data = (data - min) / (max - min).
        - This function works with both numpy arrays and PyTorch tensors, automatically determining the type 
          of the input.
        - If the input data has constant values (where max equals min), the normalized result will be all zeros.
    """
    
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

def visualize_recon_progress(recon_progress, init_data, obs_data, sensitivity_iter, urban_iter, window, save_path):
    """
    Visualizes the reconstruction progress by plotting the initial, reconstructed, and observed images.

    Args:
        recon_progress (list): A list of tensors containing the reconstructed images at each iteration. 
                               Each tensor should have the shape (B, C, Z, H, W), where:
                               - B is the batch size,
                               - C is the number of channels,
                               - Z is the number of slices or depths,
                               - H is the height of the image,
                               - W is the width of the image.
        init_data (torch.Tensor or numpy.ndarray): The initial condition data used for reconstruction, 
                                                   with the same shape as the data in `recon_progress`.
        obs_data (torch.Tensor or numpy.ndarray): The observed data used for comparison, with the same 
                                                  shape as the data in `recon_progress`.
        sensitivity_iter (int): The current iteration index for sensitivity analysis, used in the output filename.
        urban_iter (int): The current iteration index for urban modeling, used in the output filename.
        window (int): The current window index, used in the output filename.
        save_path (str): The directory path where the output images will be saved. The directory will be created if it doesn't exist.

    Returns:
        None: The function saves the generated plots as PNG files in the specified directory.

    Notes:
        - The function creates a grid of subplots to display the initial, reconstructed, and observed images 
          for a selected set of random indices. 
        - It assumes the presence of 3 channels in the data (e.g., RGB images) and visualizes the images 
          for the specified random indices across multiple iterations.
        - Each generated plot is saved in the directory with a filename that includes the 
          window, sensitivity iteration, urban iteration, and channel index.
        - If the specified directory does not exist, it will be created automatically.
    """

    image_indices = [1, 2, 3, 4, 20]
    num_channels = 3

    for channel in range(num_channels):
        # Plot original and reconstructed images
        fig, axes = plt.subplots(len(recon_progress)+2, len(image_indices), figsize=(20, 20))

        # initial condition
        for i, idx in enumerate(image_indices):
            if torch.is_tensor(init_data):
                axes[0, i].imshow(init_data[0, channel, idx, :, :].cpu().detach().numpy())
            else:
                axes[0, i].imshow(init_data[0, channel, idx, :, :])
            axes[0, i].set_title(f'Init z = {idx + 1}')
            axes[0, i].axis('off')

        for iter in range(len(recon_progress)):
            for i, idx in enumerate(image_indices):
                reconstructed = recon_progress[iter][0, channel, idx, :, :].cpu().detach().numpy()

                # Plot reconstructed image
                axes[iter + 1, i].imshow(reconstructed)
                axes[iter + 1, i].set_title(f'Iter = {iter}, z = {idx + 1}')
                axes[iter + 1, i].axis('off')

        # last row is the observation data
        for i, idx in enumerate(image_indices):
            if torch.is_tensor(obs_data):
                axes[-1, i].imshow(obs_data[0, channel, idx, :, :].cpu().detach().numpy())
            else:
                axes[-1, i].imshow(obs_data[0, channel, idx, :, :])
            axes[-1, i].set_title(f'OBS z = {idx + 1}')
            axes[-1, i].axis('off')

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(save_path, f"w{window}s{sensitivity_iter}u{urban_iter}c{channel}.png"))
        print("Image saved")

def checkpointed_forward(model, *inputs):
    """
    Performs a forward pass through the model using gradient checkpointing to save memory.

    Args:
        model (torch.nn.Module): The model to be evaluated, which should support the forward pass.
        *inputs: Variable length argument list representing the input tensors to the model. 
                  These tensors should be compatible with the model's forward method.

    Returns:
        torch.Tensor: The output of the model after the forward pass.

    Notes:
        - Gradient checkpointing allows for trading compute for memory by re-computing 
          intermediate activations during the backward pass instead of storing them.
        - This function uses the `checkpoint` function from `torch.utils.checkpoint`, 
          which allows for memory-efficient training of large models.
        - Ensure that the model and inputs are on the same device (CPU or GPU) to avoid 
          runtime errors during evaluation.
    """
    return checkpoint(model, *inputs)

def create_pollution_source(map, start_row, end_row, start_col, end_col, value, slope_row=1, slope_col=1, thickness=2, device):
    row = start_row
    col = start_col

    while row < end_row and col < end_col:
        for i in range(-thickness, thickness + 1):
            for j in range(-thickness, thickness + 1):
                if (0 <= row + i < map.shape[3]) and (0 <= col + j < map.shape[4]):
                    map[0, 0, 1:2, row + i, col + j] = torch.tensor(value, dtype=torch.float32, device=device)
        row += slope_row
        col += slope_col

    return map