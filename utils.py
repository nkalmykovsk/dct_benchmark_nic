import subprocess
import torch
import sys
import os
import struct
import math
import pickle
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import compressai.zoo as compressai_models
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from skimage.morphology import opening, dilation, square
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Get the base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_gpu_memory_and_temperature():
    """
    Retrieves memory usage and temperature for all available GPUs.
    Returns a dictionary with GPU ID as the key and a tuple of (memory used, temperature).
    """
    try:
        # Get memory and temperature data
        result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'],
        encoding='utf-8'
        )
        # Parse GPU data: {GPU_ID: (free_memory, temperature)}
        gpu_data = {
            int(line.split(',')[0]): (int(line.split(',')[1]), int(line.split(',')[2]))
            for line in result.strip().split('\n')
        }
        
        return gpu_data
    except Exception as e:
        print(f"Error retrieving GPU information: {e}")
        return {}

def select_gpu(gpu_order, memory_threshold=8000, temp_threshold=70):
    """
    Selects a GPU based on the specified order, memory, and temperature thresholds.
    
    Args:
        gpu_order (list): List of GPU IDs in priority order.
        memory_threshold (int): Minimum free memory in MB to qualify for selection.
        temp_threshold (int): Maximum temperature in Celsius to qualify for selection.
        
    Returns:
        int: The GPU ID of the selected GPU, or None if no GPUs meet the criteria.
    """
    gpu_data = get_gpu_memory_and_temperature()
    if not gpu_data:
        print("No GPUs detected or error reading GPU info.")
        return None

    for gpu_id in gpu_order:
        if gpu_id in gpu_data:
            memory_free, temperature = gpu_data[gpu_id]
            print(f"GPU {gpu_id}: Memory Free = {memory_free} MB, Temperature = {temperature}°C")
            if memory_free > memory_threshold and temperature < temp_threshold:
                print(f"Selected GPU: {gpu_id} (Memory Free: {memory_free} MB, Temperature: {temperature}°C)")
                return gpu_id

    print("No GPUs meet the criteria.")
    return None

def get_device():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        gpu_priority_order = [1, 2, 3]
        selected_gpu = select_gpu(gpu_order=gpu_priority_order, memory_threshold=14000, temp_threshold=70)
        
        if selected_gpu is not None:
            print(f"Running task on GPU-{selected_gpu}")
            device = torch.device(f"cuda:{selected_gpu}")
            print(f"Device set to: {device}")
            return device
        else:
            print("No suitable GPU found, falling back to CPU")
            return torch.device("cpu")
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")


def vif(input, target):
    """
    Visual Information Fidelity via torchmetrics.
    VIF values range from 0 to 1, where 1 indicates perfect accuracy.
    """
    from torchmetrics.image import VisualInformationFidelity
    device = get_device()
    vif_metric = VisualInformationFidelity().to(device)
    return vif_metric(input, target)


def compute_psnr(original_image, compressed_image):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images.
    Higher PSNR values indicate better image quality.
    PSNR value in decibels (dB)
    """
    import math as _math
    mean_squared_error = torch.mean((original_image - compressed_image)**2).item()
    psnr_value = -10 * _math.log10(mean_squared_error)
    return psnr_value


def ssimloss(input, target, window_size=11):
    """
    Computes 1 - Structural Similarity Index Measure (SSIM)
    Evaluates visual quality by comparing structural similarities.
    - window_size: Size of window for computing local statistics
    (default 11x11).
    """
    from kornia.losses import SSIMLoss
    loss_module = SSIMLoss(window_size=window_size)
    return loss_module(input, target)


def compute_msssim(original_image, compressed_image):
    """
    Computes Multi-Scale Structural Similarity Index
    Higher values indicate better image quality.
    data_range=1.0 indicates pixel values are normalized to [0,1]
    Conversion to dB: -10*log10(1-SSIM)
    """
    import math as _math
    from pytorch_msssim import ms_ssim
    ssim_score = ms_ssim(original_image, compressed_image, data_range=1.).item()
    return -10 * _math.log10(1 - ssim_score)


def compute_bpp(network_output):
    """
    Computes bits per pixel (BPP) for the compressed image.
    Lower BPP means stronger compression.
    Higher BPP means better quality but larger file size.
    
    Args:
        network_output: dictionary with output data:
            - x_hat: reconstructed image
            - likelihoods: probabilities for entropy coding
    """
    import math as _math
    # (batch, channels, height, width)
    image_size = network_output['x_hat'].size()
    
    # Calculate total number of pixels
    total_pixels = image_size[0] * image_size[2] * image_size[3]
    
    # Calculate BPP as sum of log probabilities divided by number of pixels
    bits_per_pixel = sum(
        torch.log(prob).sum() / (-_math.log(2) * total_pixels)
        for prob in network_output['likelihoods'].values()
    ).item()
    return bits_per_pixel


def dists(input, target):
    """
    Computes the DISTS metric between input and target images
    """
    import piq
    dists_metric = piq.DISTS(reduction="none")
    return dists_metric(input, target)


def psnr(input, target):
    """
    PSNR metric using kornia PSNRLoss with max_val=1.0
    """
    from kornia.losses import PSNRLoss
    psnr_metric = PSNRLoss(max_val=1.0)
    return psnr_metric(input, target)


def write_uints(fd, values, fmt=">{:d}I"):
    """
    Function to write unsigned integers
    Returns number of bytes written (4 bytes per number)
    """
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def write_uchars(fd, values, fmt=">{:d}B"):
    """
    Function to write unsigned chars
    Returns number of bytes written (1 byte per char)
    """
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_body(fd, shape, out_strings):
    """
    Function to write the main content of the compressed file
    """
    bytes_cnt = 0
    # Write dimensions and number of strings
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    # Write each data string
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt  # Returns total number of bytes written


def filesize(filepath: str) -> int:
    """
    Gets file size in bytes
    """
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_bytes(fd, values, fmt=">{:d}s"):
    """
    Writes byte data to file
    """
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1

def savecompressed(compressfile, outnet, bitdepth, h, w):
    """
    Saves compressed image to file
    """
    shape = outnet["shape"]
    with Path(compressfile).open("wb") as f:
        write_uints(f, (h, w))           # Write dimensions
        write_uchars(f, (bitdepth,))     # Write bit depth
        write_body(f, shape, outnet["strings"])  # Write compressed data
    size = filesize(compressfile)
    bpp = float(size) * 8 / (h * w)  # Calculate bits per pixel
    return bpp

def save_checkpoint(state, filename="checkpoint.pkl"):
    """
    Saves model state to file
    """
    with open(filename, "wb") as f:
        pickle.dump(state, f)


def bpp_loss(output, num_pixels):
    """
    Calculates bits per pixel for compressed image
    """
    bpp = sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    return bpp

"""
Functions for evaluating image compression model performance.

eval_perf() - basic evaluation function that calculates:
- PSNR (Peak Signal-to-Noise Ratio) between original and reconstructed image
- Bpp (Bits per pixel) - theoretical and actual compressed file size
- SSIM (Structural Similarity Index) for visual quality assessment

eval_perf_full() - extended version that additionally calculates:
- PSNR between original image and perturbed input
- VIF (Visual Information Fidelity) for assessing visual information preservation
"""

# Calculate baseline peformance for the original output
def eval_perf(model, original_image, img_path):
    """
    Evaluates basic compression model performance.
    
    Args:
        model: compression model (on GPU)
        original_image: original input image
        img_path: path to image file
    
    Returns:
        dict with metrics: PSNR, Bpp, Bpp(fsize), SSIM
    """
    MAX_PIXEL_VALUE = 1
    num_pixels = original_image.shape[2] * original_image.shape[3]

    with torch.no_grad():
        # Get reconstructed image
        model_output = model.forward(original_image) # dict_keys(['x_hat', 'likelihoods'])
        reconstructed_image = model_output["x_hat"].clamp_(0, 1) 
        
        # Calculate PSNR: higher mse means lower psnr and worse image quality
        mse_loss = F.mse_loss(reconstructed_image, original_image)
        psnr_value = 10 * torch.log10((MAX_PIXEL_VALUE**2) / (mse_loss))

        # Calculate theoretical Bpp. There are two latent code streams: y and z.
        # y is the main image representation, z is hyperparameters (hyper-prior)
        # that help better compress y
        bpp_theoretical = (
            torch.log(model_output["likelihoods"]["y"]).sum()
            + torch.log(model_output["likelihoods"]["z"]).sum()
        ) / (-math.log(2) * num_pixels)
        # Theoretical: since this is estimate before actual arithmetic coder

        # Calculate SSIM
        ssimloss_value = ssimloss(reconstructed_image, original_image)

        # Compress the image
        compressed_data = model.compress(original_image)
        
        bytes_y = sum(len(s) for s in compressed_data['strings'][0])
        bytes_z = sum(len(s) for s in compressed_data['strings'][1])
        size_bytes = bytes_y + bytes_z
        raw_bytes = original_image.shape[2] * original_image.shape[3] * original_image.shape[1]  # 8-bit RGB
        compression_ratio = raw_bytes / size_bytes
        print(f'Compression ratio: {compression_ratio:.1f}x')

    # Create unique name for compressed file
    unique_id = np.random.randint(1000, 9999)
    compressed_filename = os.path.splitext(img_path)[0] + "compress" + str(unique_id)

    # Save compressed file and get actual Bpp
    height, width = original_image.size(2), original_image.size(3)
    bpp_actual = savecompressed(compressed_filename, compressed_data, bitdepth=8, h=height, w=width)

    return {
        "PSNR": psnr_value.cpu().detach().numpy(),
        "Bpp": bpp_theoretical.cpu().detach().numpy(),
        "Bpp(fsize)": bpp_actual,
        "SSIMLoss": ssimloss_value.cpu().detach().numpy(),
    }


def eval_perf_full(model, perturbed_image, original_image, img_path):
    """
    Extended performance evaluation with additional metrics.
    
    Args:
        model: compression model (on GPU)
        perturbed_image: perturbed input image
        original_image: original image
        img_path: path to image file
    
    Returns:
        dict with extended set of metrics
    """

    MAX_PIXEL_VALUE = 1
    num_pixels = perturbed_image.shape[2] * perturbed_image.shape[3]

    with torch.no_grad():
        # Get reconstructed image from perturbed input
        model_output = model.forward(perturbed_image)
        reconstructed_image = model_output["x_hat"].clamp_(0, 1)
        
        # PSNR between reconstructed and perturbed
        mse_loss = F.mse_loss(reconstructed_image, perturbed_image)
        psnr_ao = 10 * torch.log10((MAX_PIXEL_VALUE**2) / mse_loss)

        # Theoretical Bpp
        bpp_theoretical = (
            torch.log(model_output["likelihoods"]["y"]).sum()
            + torch.log(model_output["likelihoods"]["z"]).sum()
        ) / (-math.log(2) * num_pixels)

        # SSIM between reconstructed and original
        ssimloss_value = ssimloss(reconstructed_image, original_image)

        # Compress perturbed image
        compressed_data = model.compress(perturbed_image)

    # Save the compressed file
    unique_id = np.random.randint(1000, 9999)
    compressed_filename = os.path.splitext(img_path)[0] + "compress" + str(unique_id)
    height, width = perturbed_image.size(2), perturbed_image.size(3)
    bpp_actual = savecompressed(compressed_filename, compressed_data, bitdepth=8, h=height, w=width)
    
    # Calculate VIF metrics
    vif_score_in = vif(original_image, perturbed_image)
    vif_score_out = vif(original_image, reconstructed_image)
    
    # PSNR between original and perturbed
    mse_perturbed_original = F.mse_loss(original_image, perturbed_image)
    psnr_ai_oi = 10 * torch.log10((MAX_PIXEL_VALUE ** 2) / mse_perturbed_original)

    return {
        "PSNR(ai,ao)": psnr_ao.cpu().detach().numpy(),  # PSNR between perturbed and reconstructed
        "PSNR(ai,oi)": psnr_ai_oi.cpu().detach().numpy(),  # PSNR between perturbed and original
        "Bpp": bpp_theoretical.cpu().detach().numpy(),
        "Bpp(fsize)": bpp_actual,
        "SSIMLoss(ao)": ssimloss_value.cpu().detach().numpy(),
        "VIF(ai,oi)": vif_score_in.cpu().detach().numpy(),  # VIF between perturbed and original
        "VIF(ao,oi)": vif_score_out.cpu().detach().numpy(),  # VIF between reconstructed and original
    }

def visualize_all_wavelet_details(details, channel=0, cmap='bwr'):
    """
    Visualize wavelet coefficients at all scales and bands for a given channel.

    Args:
        details (list of torch.Tensor): list of tensors of shape [1, C, 3, H, W]
        channel (int): which input channel to visualize (e.g., 0 for R or grayscale)
        cmap (str): colormap for display
    """
    num_scales = len(details)
    num_bands = 3  # LH, HL, HH

    fig, axes = plt.subplots(num_scales, num_bands, figsize=(num_bands * 4, num_scales * 4))
    if num_scales == 1:
        axes = axes[None, :]  # ensure 2D indexing if only one scale

    for scale_idx, detail in enumerate(details):
        for band_idx in range(num_bands):
            ax = axes[scale_idx, band_idx]
            coeff = detail[0, channel, band_idx].detach().cpu().numpy()
            coeff /= (np.max(np.abs(coeff)) + 1e-8)  # normalize for display
            
            vmax = np.max(np.abs(coeff))
            ax.imshow(coeff, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(f"Scale {scale_idx+1}, Band {['LH','HL','HH'][band_idx]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


 
#---------------------------------------------
def maxdistortion_total(x, errbound=0.1, smoothfilter = None, qualitymeasure = 'psnr', 
                         target_quality=None,quality_loss_lambda=0.1,l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,\
                        keep_perturbation_targeted = False,keep_low_outcomequality = True,
               noise_projection='clamp',noisemapping = 'additive',keep_minimum_attack_area= True,keep_maximal_defect_area = False, clamp_input = False,clamp_output = False):
    '''
    minimize_n:  Quality + lambda * (AttackRegion - DefectRegion)
    subject to:  x_out = f(x_pert), with |n_ij| <= delta,
            
    where:
        x_pert = x + n: the perturbed input,

        x_out = f(x_pert): the output after processing,

        delta: the max allowed perturbation per pixel.
    
    Quality |PSNR(x_out, x_ori) - PSNR_target_out| and |PSNR(x_pert, x_ori) - PSNR_target_in|
        control distortion in both the output and the perturbed input.
        
    AttackRegion:  ||n||_inf,1 (l_inf-l1 norm) encouraging sparse, localized noise
        
    DefectRegion: ||x_out - x_pert||_inf,1   encouraging the perturbation to induce stronger effects on the output image.
        
        
    The objective seeks an adversarial perturbation n that:
    
    The PSNR-based Quality term can be replaced with other quality metrics (e.g., SSIM, VIF, DIST), with target performance typically set to zero.
    ''' 
    # x (torch.Tensor): Input image of size (1, C, H, W).
    # errbound (float): Noise bound value.
    # smoothfilter: Optional smoothing filter (e.g., Gaussian).
    # qualitymeasure (str): 'psnr', 'mse', 'ssim', or 'dists'.
    # target_quality (int or list): Target quality value(s). If int, duplicated.
    # quality_loss_lambda (float): Weight for quality loss.
    # l1_lambda (float): Weight for L1 regularization.
    # num_iterations (int): Number of optimization steps.
    # model (torch.nn.Module): Neural network model.
    # device (torch.device): Device to run the optimization.
    # mask (torch.Tensor): Optional binary mask for applying noise.
    # initial_noise (torch.Tensor): Optional initial noise.
    # learningrate (float): Learning rate.
    # keep_perturbation_targeted (bool): Enforce target quality for perturbed input.
    # keep_low_outcomequality (bool): Minimize quality above target.
    # noise_projection (str): 'clamp', 'smartclamp', 'tanh', 'smoothell1'.
    # noisemapping (str): 'additive', 'multiplicative', 'logexp', 'tanhatanh'.
    # keep_minimum_attack_area (bool): Encourage sparse perturbations.
    # keep_maximal_defect_area (bool): Encourage wider distortion area.
    # clamp_input (bool): Clamp perturbed input between 0 and 1.
    # clamp_output (bool): Clamp model output between 0 and 1.
     
    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if target_quality is None:
        target_quality = [0, 0]
        
 
    # Check if it's an integer
    if isinstance(target_quality, int):
        # Convert it to a list with two identical elements
        target_quality = [target_quality, target_quality]
    
    
    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * 0.01*torch.randn_like(x) * mask).to(device)
        
    else:
        noise_pattern = torch.nn.Parameter(initial_noise).to(device) 

    #     print(single_channel_noise.shape)
    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.Adam([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3] 
    
    # Clamp the attacked image to [0,1]
    # clamp_input = True
    
    
    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
        
             # --- Noise projection --- 
            if noise_projection == 'clamp':
                noise_pattern.data.clamp_(-errbound, errbound)
                noise_pattern2 = noise_pattern
                
            elif noise_projection == 'smartclamp':
                # noise_pattern.data.clamp_(-errbound, errbound)
                # noise_pattern2 = noise_pattern
                noise_pattern2 = SmartClamp.apply(noise_pattern,-errbound, errbound)            
                
            elif noise_projection == 'tanh':
                noise_pattern2 = errbound*torch.tanh(noise_pattern)
 
            elif noise_projection == 'smoothell1':
                noise_pattern2 = errbound*(noise_pattern/torch.sqrt(1+noise_pattern**2))

            else:
                raise ValueError(f"Unsupported noise_projection: {noise_projection}")
            
    
            # --- Optional smoothing --- 
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask 
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
                
            
            # --- Noise mapping ---
            if noisemapping == 'logexp':
                perturbed_image = torch.log(torch.exp(x)+smoothed_noise_pattern)
            elif noisemapping == 'tanhatanh':
                perturbed_image = torch.tanh(torch.atanh(x)+smoothed_noise_pattern)
                
            elif noisemapping == 'gamma':
                perturbed_image = x**(1+smoothed_noise_pattern)

            elif noisemapping == 'sigmoid':
                perturbed_image = x + smoothed_noise_pattern * x*(1-x)

            elif noisemapping == 'multiplicative' :          
                perturbed_image = x + smoothed_noise_pattern * torch.exp(-x)
            elif noisemapping == 'additive' :          
                perturbed_image = x + smoothed_noise_pattern
                
            #print(perturbed_image)
            
    
            # Forward pass through the model
            if clamp_input == True:
                output = model.forward(torch.clamp(perturbed_image,0,1))
            else:
                output = model.forward(perturbed_image)
            
    
            # Extract reconstruction
            if clamp_output == True:
                perturbed_output = torch.clamp(output['x_hat'],0,1)
            else:
                perturbed_output = output['x_hat']
                
 
                
                
            # Calculate MSE loss(perturbed_output, x) : keep the decompressed image close to the original image
            mse_loss = F.mse_loss(perturbed_output, x)

            # Calculate PSNR loss
            perturbed_quality = 10 * torch.log10((MAX_I ** 2) / mse_loss)
            
            # Compute the difference in PSNR between perturbed and target
            if keep_low_outcomequality==True:
            # print('perturbed_quality - target_quality[1]')
                quality_loss = torch.max((perturbed_quality - target_quality[1]).abs(),torch.tensor(0.0))
            else:
                quality_loss = perturbed_quality 
                    

            if keep_perturbation_targeted == True:
                # print('perturbed_quality_ai - target_quality[0]')
                # Calculate MSE loss(perturbed_output, x) : keep the decompressed image close to the original image
                mse_loss_ai = F.mse_loss(perturbed_image, x)

                # Calculate PSNR loss
                perturbed_quality_ai = 10 * torch.log10((MAX_I ** 2) / mse_loss_ai)

                # Compute the difference in PSNR between perturbed and target
                # quality_loss = torch.max((perturbed_quality - target_quality).abs()
                quality_loss_ai = torch.max((perturbed_quality_ai - target_quality[0]).abs()-2,torch.tensor(0.0))
                
            else:
                perturbed_quality_ai = 0
                quality_loss_ai=0
                                        
                                     
            # --- Total loss ---
            combined_loss = (quality_loss+quality_loss_ai)  
            
            # Perform gradient descent
            combined_loss.backward()
             
                
            
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:   
                
                print(f'Iteration {iteration} | {qualitymeasure}: (ao,oi) {perturbed_quality:.4f} - Lost {quality_loss: .4f} | (ai,oi) {perturbed_quality_ai:.4f} Lost {quality_loss_ai:.4f} | Loss {combined_loss : .4f}')
 
    except KeyboardInterrupt:
        #  
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


def gaussian_kernel(size, sigma):
    # Create a vector of size 'size' with values from -size//2 to size//2
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    # Calculate the Gaussian distribution for each value in the vector
    g = torch.exp(-(x**2) / (2 * sigma**2))
    # Normalize the distribution so it sums to 1
    g /= g.sum()
    # Create a 2D Gaussian kernel from the outer product of the vector with itself
    return g.outer(g)


def allocate_attack_mask(perturbed_image,perturbed_output):
    # Allocation the effective noise region 
    # Anh-Huy Phan
    #
    device = perturbed_image.device
    # Gaussian blur to the mask to smooth the edges
    kernel_size = 21
    sigma_filter = 8

    # Create the Gaussian kernel
    gaussian_filter_mask = gaussian_kernel(kernel_size, sigma_filter)

    # Add batch and channel dimensions to the filter
    gaussian_filter_mask = gaussian_filter_mask.view(1, 1, *gaussian_filter_mask.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter_mask = gaussian_filter_mask.repeat(perturbed_image.size(1), 1, 1, 1)
    gaussian_filter_mask = gaussian_filter_mask.to(device)


    # 
    residue = perturbed_output-perturbed_image
    # residue[torch.abs(residue)>0.5] = 1
    # residue =
    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])
    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])

    ell2_noise = torch.sqrt(torch.sum(residue**2,dim = 1));

    mask_noise = ell2_noise>.5#torch.max(ell2_noise) * 1e-1
    mask_noise = mask_noise.squeeze()

    mask_noise = mask_noise.cpu().detach().numpy()
    mask_noise = opening(mask_noise, square(10))
    # mask_noise = opening(mask_noise, square(10))
    mask_noise = dilation(mask_noise, square(20))
    # mask_noise = dilation(mask_noise, square(5))

    # plt.imshow(mask_noise)

    # Mask 3D 
    new_mask = torch.zeros_like(perturbed_image)
    nnz_ix = np.where(mask_noise==1)
    new_mask[:,:,nnz_ix[0],nnz_ix[1]] = 1

    # new_mask,new_2dmask = noise_to_mask(residue)
    new_mask = F.conv2d(new_mask, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])

    return new_mask


def get_entropy_mask(img, model_name, quality, device, disk_radius=10):
    """Generate an entropy-based mask for the given image and model configuration."""
    # Convert to grayscale
    gray_img = rgb2gray(img)

    # Compute entropy image
    entr_img = entropy(gray_img, disk(disk_radius))

    # Define entropy weight (k_entropy) based on model and quality
    entropy_weights = {
        'cheng2020-attn': {5: 3, 6: 4},
        'cheng2020-anchor': {5: 2, 6: 2},
        'tcm': {5: 4, 6: 5}
    }

    # Validate model and quality
    if model_name not in entropy_weights:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(entropy_weights.keys())}")
    if quality not in entropy_weights[model_name]:
        raise ValueError(f"Unsupported quality ({quality}) for model {model_name}. "
                         f"Supported qualities: {list(entropy_weights[model_name].keys())}")

    k_entropy = entropy_weights[model_name][quality]

    # Compute entropy mask
    entropy_mask = torch.tensor(entr_img, device=device, dtype=torch.float)
    entropy_mask = torch.exp(entropy_mask / k_entropy).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    
    # Normalize mask
    entropy_mask /= entropy_mask.ravel().max()

    return entropy_mask, entr_img


def evaluate_frequency_response(model, device, size=64):
    """
    Evaluate frequency distortion of a NIC model using a 3-channel DCT basis matrix image.

    Args:
        model: NIC model with interface model(x)['x_hat']
        device: torch device ("cuda" or "cpu")
        size: size of the DCT matrix image (default 64x64)
    """
    # 1. Generate DCT basis matrix image (grayscale)
    x = np.eye(size)  # identity
    x_dct = dct(x, axis=1, norm="ortho")

    # replicate into 3 channels (RGB-like)
    x_dct_rgb = np.stack([x_dct, x_dct, x_dct], axis=-1)

    # normalize for model input [0,1]
    mx, Mx = x_dct_rgb.min(), x_dct_rgb.max()
    x_dct_norm = (x_dct_rgb - mx) / (Mx - mx)

    # to tensor [1,3,H,W]
    x_tensor = torch.from_numpy(x_dct_norm).float().permute(2,0,1).unsqueeze(0).to(device)

    # 2. Compress & decompress using NIC model
    with torch.no_grad():
        out = model(x_tensor)
        x_hat_norm = out['x_hat'].cpu()

    # denormalize back
    x_hat = mx + (Mx - mx) * x_hat_norm.squeeze().permute(1,2,0).numpy()

    # 3. Apply inverse DCT channel-wise
    x_idct = np.zeros_like(x_hat)
    for c in range(3):
        x_idct[...,c] = idct(x_hat[...,c], axis=1, norm="ortho")

    # 4. Compute diagonal distortion (average across channels)
    diag_recon = np.mean([np.diag(x_idct[...,c]) for c in range(3)], axis=0)
    true_diag = np.ones_like(diag_recon)
    diag_mse = np.mean((diag_recon - true_diag) ** 2)

    # bandwise errors (split diagonal into 3 parts)
    n = len(diag_recon)
    bands = np.array_split(np.arange(n), 3)
    band_errors = [np.mean((diag_recon[idx] - 1) ** 2) for idx in bands]

    # 5. Visualization
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(x_dct_norm, cmap="gray")
    axs[0].set_title("Original DCT matrix (RGB)")

    axs[1].imshow(x_hat_norm.squeeze().permute(1,2,0).numpy(), cmap="gray")
    axs[1].set_title("Decompressed DCT (RGB)")

    axs[2].imshow(x_idct, cmap="gray");
    axs[2].set_title("iDCT of decompressed (RGB)")

    axs[3].plot(diag_recon, label="Reconstructed Diagonal")
    axs[3].plot(true_diag, "--", label="True Identity")
    axs[3].set_title("Frequency Distortion Curve")
    axs[3].legend()

    plt.show()

    print("Diagonal MSE (overall frequency distortion):", diag_mse)
    print("Bandwise errors [low, mid, high]:", band_errors)

    return diag_mse, band_errors


def compute_dct_smearing_metrics(D_hat, axis=0, normalize_freq=True, show_plots=True, title_prefix="", save_path=None):
    """
    D_hat: (N,N) numpy array — the decompressed DCT matrix image.
           (If you have RGB, convert to gray or take one channel.)
    axis: 0 or 1: which axis of D_hat corresponds to basis vectors (default 0 -> columns are basis vectors).
          If columns are basis vectors, use axis=1 for dct along column? See usage below.
    normalize_freq: if True, report centroids normalized to [0,1].
    Returns: dict with metrics and the full frequency-response matrix R (shape N x N).
    """
    # Ensure 2D square
    assert D_hat.ndim == 2 and D_hat.shape[0] == D_hat.shape[1], "D_hat must be square 2D array"
    N = D_hat.shape[0]

    # If columns are basis vectors, we'll extract d_k = D_hat[:, k]
    # We must be consistent with how original D was built (dct applied along rows or columns).
    # Here we assume original D was produced by applying 1D DCT along axis=1 to identity,
    # so columns of D are basis vectors in spatial domain -> use axis=0 to extract columns.
    # Compute DCT of each reconstructed basis vector to get frequency coefficients.
    R = np.zeros((N, N), dtype=float)  # R[i,k] = power at frequency i for input basis k

    for k in range(N):
        if axis == 0:
            d_k_hat = D_hat[:, k].astype(float)   # vector of length N
        else:
            d_k_hat = D_hat[k, :].astype(float)

        # compute DCT coefficients of reconstructed spatial vector
        c = dct(d_k_hat, norm='ortho')   # length N
        power = np.abs(c)**2
        total = power.sum()
        if total == 0:
            p = np.zeros_like(power)
        else:
            p = power / total
        R[:, k] = p

    # Metrics per basis k
    indices = np.arange(N)
    centroids = (R.T @ indices)  # shape (N,), centroid in index units
    # Absolute centroid shift Δ_k = μ_k - k
    centroid_shift = centroids - indices

    if normalize_freq:
        # Normalize centroid positions to [0,1]
        centroids_norm = centroids / (N - 1)
        # Normalize shift by maximum possible shift for that k
        max_shift = np.maximum(indices, (N - 1) - indices)
        centroid_shift_norm = centroid_shift / (max_shift + 1e-12)
    else:
        centroids_norm = centroids
        centroid_shift_norm = centroid_shift

    # Linear leakage (1 - p_k)
    leakage = (1.0 - np.diag(R))

    # Stable ODR with tanh squashing
    eps = 1e-12
    diag_vals = np.diag(R)
    odr = (np.sum(R, axis=0) - diag_vals) / (diag_vals + eps)
    odr = np.tanh(0.5 * odr)

    # variance (spread)
    variance = np.array([np.sum(((indices - centroids[k])**2) * R[:, k]) for k in range(N)])
    spread = np.sqrt(variance)
    # Normalize spread by max possible spread from centroid
    max_spread_per_k = np.maximum(np.abs(0 - centroids), np.abs((N - 1) - centroids))
    spread = spread / (max_spread_per_k + 1e-12)

    # entropy normalized to [0,1]
    n = R.shape[0]
    entropy = -np.sum(R * np.log(R + eps), axis=0)
    entropy = entropy / np.log(n)

    # cumulative energy within windows
    def cumulative_energy(k, w):
        lo = max(0, k - w)
        hi = min(N - 1, k + w)
        return R[lo:hi+1, k].sum()

    # example windows: 0 (exact), 1,2,4 neighbors
    windows = [0, 1, 2, 4, 8]
    cum_energy = {w: np.array([cumulative_energy(k, w) for k in range(N)]) for w in windows}

    metrics = {
        'R': R,
        'leakage': leakage,
        'odr': odr,
        'centroids': centroids_norm,
        'centroid_shift': centroid_shift_norm,
        'spread': spread,
        'entropy': entropy,
        'cum_energy': cum_energy,
        'indices': indices
    }

    if show_plots:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # heatmap R
        ax = axs[0, 0]
        im = ax.imshow(R, origin='lower', cmap='viridis', interpolation='nearest')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label='Normalized power')
        ax.set_xlabel('input basis k')
        ax.set_ylabel('observed frequency i')
        ax.set_title(f'{title_prefix} Frequency-response matrix R')

        # leakage, centroid shift, spread, entropy
        ax = axs[0, 1]
        ax.plot(indices, leakage, label='Leakage (1 - p_k)')
        ax.plot(indices, np.abs(centroid_shift_norm), label='|Centroid shift| (normalized)')
        ax.plot(indices, spread, label='Spread (normalized)')
        ax.plot(indices, entropy, label='Entropy (normalized)')
        ax.set_xlabel('basis k (normalized freq index)')
        ax.legend()
        ax.set_title(f'{title_prefix} Leakage / shift / spread / Entropy')

        # ODR only
        ax = axs[1, 0]
        ax.plot(indices, odr, label='ODR (tanh scaled)')
        ax.set_xlabel('basis k')
        ax.legend()
        ax.set_title(f'{title_prefix} Off-diag ratio')

        # cumulative energy
        ax = axs[1, 1]
        for w in windows:
            ax.plot(indices, cum_energy[w], label=f'within +/-{w}')
        ax.set_xlabel('basis k')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title(f'{title_prefix} Cumulative energy in neighbor windows')

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        plt.show()

    return metrics

# ---------------------------------------------------------
# Band helpers for concise trend reporting
# ---------------------------------------------------------
def make_bands(size, mode="thirds"):
    """
    Create low/mid/high frequency index bands for a given DCT size.
    Currently uses approximately equal contiguous thirds of indices [0..size-1].
    Returns a dict with numpy index arrays: {"low": idxs, "mid": idxs, "high": idxs}.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if mode != "thirds":
        raise ValueError(f"Unsupported band mode: {mode}")
    one_third = size // 3
    two_third = 2 * size // 3
    bands = {
        "low": np.arange(0, one_third, dtype=int),
        "mid": np.arange(one_third, two_third, dtype=int),
        "high": np.arange(two_third, size, dtype=int),
    }
    # Ensure no empty bands for typical sizes; for pathological small sizes, bands may be empty
    return bands


def compute_band_summaries(
    R_mean,
    odr_avg,
    centroid_shift_avg,
    spread_avg,
    entropy_avg,
    cum_energy_avg,
    ce_window=2,
    normalize=True,
):
    """
    Summarize key metrics over low/mid/high frequency bands.

    Returns a dict:
      {
        "low": {"L_k": ..., "ODR_k": ..., "|Delta_c_k|_norm": ..., "s_k_norm": ..., "H_k_bits": ..., "CE_k(w=2)": ...},
        "mid": {...},
        "high": {...}
      }
    """
    N = R_mean.shape[0]
    diag_R = np.diag(R_mean)
    L_linear = 1.0 - diag_R  # linear leakage per k
    # Recompute entropy per k in bits from R_mean to ensure consistent units
    eps = 1e-12
    H_bits = -np.sum(R_mean * np.log(R_mean + eps), axis=0) / np.log(2)
    if ce_window not in cum_energy_avg:
        raise ValueError(f"Requested CE window {ce_window} not present in cum_energy_avg")
    CE_w = cum_energy_avg[ce_window]

    denom = (N - 1) if normalize and N > 1 else 1.0

    bands = make_bands(N, mode="thirds")
    out = {}
    for name, idx in bands.items():
        if idx.size == 0:
            # Fallback to NaNs for degenerate very-small sizes
            out[name] = {
                "L_k": float("nan"),
                "ODR_k": float("nan"),
                "|Delta_c_k|_norm": float("nan"),
                "s_k_norm": float("nan"),
                "H_k_bits": float("nan"),
                f"CE_k(w={ce_window})": float("nan"),
            }
            continue
        out[name] = {
            "L_k": float(np.median(L_linear[idx])),
            "ODR_k": float(np.median(odr_avg[idx])),
            "|Delta_c_k|_norm": float(np.median(np.abs(centroid_shift_avg[idx])) / denom),
            "s_k_norm": float(np.median(spread_avg[idx]) / denom),
            "H_k_bits": float(np.median(H_bits[idx])),
            f"CE_k(w={ce_window})": float(np.median(CE_w[idx])),
        }
    return out

# ---------------------------------------------------------
# Frequency Response Evaluation for NIC models
# ---------------------------------------------------------
def evaluate_frequency_response2(
    model,
    size=64,
    device="cuda",
    show_plots=True,
    num_runs=10,
    show_metric_plots=True,
    seed=None,
):
    """
    Evaluate frequency response of a NIC model using DCT basis matrix input.

    Args:
        model: Trained NIC model (expects input in [0,1], shape [1,3,H,W])
        size: Image size (NxN)
        device: "cuda" or "cpu"
        show_plots: Whether to visualize inputs/outputs

    Returns:
        x_dct_rgb: Original DCT basis (H,W,3)
        x_hat: Decompressed DCT basis from last run (H,W,3)
        metrics: Dictionary of averaged frequency smearing metrics over num_runs
    """
    # Optional reproducibility
    if seed is not None:
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
        except Exception:
            pass
    # 1. Construct DCT matrix (apply 1D DCT to identity)
    x = np.eye(size)
    x_dct = dct(x, axis=1, norm="ortho")

    # 2. Replicate into 3 channels
    x_dct_rgb = np.stack([x_dct, x_dct, x_dct], axis=-1)

    # 3. Normalize to [0,1]
    mx, Mx = x_dct_rgb.min(), x_dct_rgb.max()
    x_dct_norm = (x_dct_rgb - mx) / (Mx - mx + 1e-9)

    # 4. Convert to tensor [1,3,H,W]
    x_tensor = torch.from_numpy(x_dct_norm).float().permute(2,0,1).unsqueeze(0).to(device)

    # 5. Run multiple forwards and aggregate metrics
    leakage_list = []
    odr_list = []
    centroid_shift_list = []
    centroids_list = []
    spread_list = []
    entropy_list = []
    cum_energy_accumulator = {}
    R_list = []

    x_hat = None
    x_idct = None
    x_hat_norm = None
    # Compress & decompress with NIC
    for run_index in range(int(max(1, num_runs))):
        with torch.no_grad():
            out = model(x_tensor)
            x_hat_norm = out["x_hat"].cpu()
        # Denormalize back to original DCT range
        x_hat_run = mx + (Mx - mx) * x_hat_norm.squeeze().permute(1,2,0).numpy()

        # Save last run's outputs for visualization
        x_hat = x_hat_run

        # iDCT per channel
        x_idct_run = np.zeros_like(x_hat_run)
        for c in range(3):
            x_idct_run[..., c] = idct(x_hat_run[..., c], axis=1, norm="ortho")
        x_idct = x_idct_run

        # Metrics on grayscale
        m = compute_dct_smearing_metrics(
            x_hat_run.mean(axis=2), axis=0, normalize_freq=True, show_plots=False, title_prefix="NIC Response"
        )
        leakage_list.append(m["leakage"])  # shape (N,)
        odr_list.append(m["odr"])          # shape (N,)
        centroid_shift_list.append(m["centroid_shift"])  # shape (N,)
        centroids_list.append(m["centroids"])            # shape (N,)
        spread_list.append(m["spread"])    # shape (N,)
        entropy_list.append(m["entropy"])  # shape (N,)
        R_list.append(m["R"])              # shape (N,N)

        # Cum energy windows
        for w_key, ce_arr in m["cum_energy"].items():
            if w_key not in cum_energy_accumulator:
                cum_energy_accumulator[w_key] = []
            cum_energy_accumulator[w_key].append(ce_arr)

    # Average across runs
    leakage_avg = np.mean(np.stack(leakage_list, axis=0), axis=0)
    odr_avg = np.mean(np.stack(odr_list, axis=0), axis=0)
    centroid_shift_avg = np.mean(np.stack(centroid_shift_list, axis=0), axis=0)
    centroids_avg = np.mean(np.stack(centroids_list, axis=0), axis=0)
    spread_avg = np.mean(np.stack(spread_list, axis=0), axis=0)
    entropy_avg = np.mean(np.stack(entropy_list, axis=0), axis=0)
    R_mean = np.mean(np.stack(R_list, axis=0), axis=0)

    cum_energy_avg = {w_key: np.mean(np.stack(arrs, axis=0), axis=0)
                      for w_key, arrs in cum_energy_accumulator.items()}

    # Build metrics dictionary: provide per-k arrays for plotting, plus compact summary for CSVs
    indices_arr = np.arange(size)
    avg_metrics = {
        "num_runs": int(max(1, num_runs)),
        # Per-k arrays for visualization
        "indices": indices_arr,
        # Linear leakage per k as defined L_k = 1 - R[k,k]
        "leakage": 1.0 - np.diag(R_mean),
        "odr": odr_avg,
        "centroid_shift": centroid_shift_avg,
        "spread": spread_avg,
        "entropy": entropy_avg,
        "cum_energy": cum_energy_avg,
        "R": R_mean,
    }

    # Aggregate scalars for table reporting (concise summary)
    summary_window = 2
    try:
        ce_w = cum_energy_avg.get(summary_window, None)
        # Compute entropy in bits from R_mean to keep CSV units consistent
        eps = 1e-12
        entropy_bits_from_R = -np.sum(R_mean * np.log(R_mean + eps), axis=0) / np.log(2)
        summary = {
            "L_k": float(np.median(1.0 - np.diag(R_mean))),
            "ODR_k": float(np.median(odr_avg)),
            "|Delta_c_k|": float(np.median(np.abs(centroid_shift_avg))),
            "s_k": float(np.median(spread_avg)),
            "H_k_bits": float(np.median(entropy_bits_from_R)),
            f"CE_k(w={summary_window})": float(np.median(ce_w)) if ce_w is not None else None,
        }
    except Exception:
        summary = None
    avg_metrics["summary"] = summary

    # 6. Visualization
    if show_plots:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(x_dct_norm, cmap="gray")
        axs[0].set_title("Original DCT matrix (RGB)")
        axs[0].axis("off")

        axs[1].imshow(x_hat_norm.squeeze().permute(1,2,0).numpy(), cmap="gray")
        axs[1].set_title("Decompressed DCT (RGB)")
        axs[1].axis("off")

        axs[2].imshow(x_idct, cmap="gray")
        axs[2].set_title("iDCT of decompressed (RGB)")
        axs[2].axis("off")
        plt.show()

    if show_metric_plots:
        indices = avg_metrics["indices"]
        fig2, axs2 = plt.subplots(2, 3, figsize=(16, 8))

        axs2[0, 0].plot(indices, avg_metrics["leakage"])
        axs2[0, 0].set_title("Leakage L_k")
        axs2[0, 0].set_xlabel("k")

        axs2[0, 1].plot(indices, avg_metrics["odr"])
        axs2[0, 1].set_title("Off–diagonal ratio ODR_k")
        axs2[0, 1].set_xlabel("k")

        axs2[0, 2].plot(indices, np.abs(avg_metrics["centroid_shift"]))
        axs2[0, 2].set_title("Centroid shift Δc_k")
        axs2[0, 2].set_xlabel("k")

        axs2[1, 0].plot(indices, avg_metrics["spread"])
        axs2[1, 0].set_title("Spread s_k")
        axs2[1, 0].set_xlabel("k")

        axs2[1, 1].plot(indices, avg_metrics["entropy"])
        axs2[1, 1].set_title("Entropy H_k")
        axs2[1, 1].set_xlabel("k")

        # CE_k(w) for several windows
        for w_key in sorted(cum_energy_avg.keys()):
            axs2[1, 2].plot(indices, cum_energy_avg[w_key], label=f"w={w_key}")
        axs2[1, 2].set_title("Cumulative energy CE_k(w)")
        axs2[1, 2].set_xlabel("k")
        axs2[1, 2].set_ylim(-0.05, 1.05)
        axs2[1, 2].legend()

        plt.tight_layout()
        plt.show()

    # Print clear one-line summary for tables
    if avg_metrics.get("summary") is not None:
        s = avg_metrics["summary"]
        print(
            f"Summary (median over k; H in bits; CE window=2; runs={avg_metrics['num_runs']}): "
            f"L_k={s['L_k']:.4f}, ODR_k={s['ODR_k']:.4f}, |Δc_k|={s['|Delta_c_k|']:.4f}, "
            f"s_k={s['s_k']:.4f}, H_k={s['H_k_bits']:.4f}, CE_k(w=2)={s['CE_k(w=2)']:.4f}"
        )

    return x_dct_rgb, x_hat, avg_metrics