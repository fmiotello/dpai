import torch
import torch.nn as nn
import sys
import numpy as np
import math
import librosa
import colorednoise as cn
# import kornia as K
import torch.nn.functional as F
import auraloss
import random
import os

#torch.manual_seed(0)

def extract_from_dataset(path, n):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.wav')]
    files = [f for f in files if librosa.get_duration(filename=os.path.join(path, f)) >= 15]
    if not len(files) >= n:
        raise Exception('Not enough files in dataset ', path)
    files = list(map(lambda file: os.path.join(path, file), files))
    files.sort()
    np.random.seed(seed=0)
    return list(np.random.choice(files, size=n, replace=False))

def generate_noise(size, beta):
    white_noise = np.random.randn(*size)
    white_noise_fft = np.fft.fftn(white_noise)

    ndims = len(size)
    freq_along_axis = []

    for axis in range(ndims):
      freq_along_axis.append(np.fft.fftfreq(size[axis]))

    grids = np.meshgrid(*freq_along_axis)
    sum_of_squares = 0

    for grid in grids:
      sum_of_squares += grid**2

    freqs = np.sqrt(sum_of_squares)
    origin = (0,) * ndims
    freqs[origin] += 1e-8      # DC component
    filter = 1/np.power(freqs, beta)

    colored_fft = white_noise_fft * filter.T
    colored_noise = np.fft.ifftn(colored_fft)

    return np.abs(colored_noise)


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'uniform':
        return x.uniform_()
    elif noise_type == 'normal':
        return x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='uniform', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        net_input = fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False

    return net_input



def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)

        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=20, verbose=True)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        for j in range(num_iter):
            optimizer.zero_grad()
            loss = closure()
            optimizer.step()
            #scheduler.step(loss)
            #print(optimizer.param_groups[0])
    else:
        assert False

def minmax_norm(X):
  diff = np.max(X) - np.min(X)
  min = np.min(X)
  return (X - min) / diff

def crop_spectrogram(spectrogram, dim_div_by):
    return spectrogram[0:(spectrogram.shape[0]//dim_div_by)*dim_div_by,0:(spectrogram.shape[1]//dim_div_by)*dim_div_by]


def nmse(original, reconstructed):
    return 10*math.log10(np.linalg.norm(reconstructed - original)**2/(np.linalg.norm(original)**2))

def nmse_torch(original, reconstructed):
    return 10*torch.log10(torch.linalg.norm(reconstructed - original)**2/(torch.linalg.norm(original)**2))

def generate_mask(input_dimensions, n_gaps, len=3): # at sr=22050 n_fft=2048 hop_length=n_fft/4 1 time frame corresponds to more or less 20ms
    # if type == 'short':
    #     len = 3
    # elif type == 'mid':
    #     len = 10
    # elif type == 'long':
    #     len = 45

    assert (input_dimensions[1] >= n_gaps*len)

    mask = np.ones(input_dimensions)
    base = input_dimensions[1]//(n_gaps + 1)

    for i in range(n_gaps):
        start_idx = (i+1)*base - len//2
        mask[:,start_idx:start_idx+len] = 0

    return mask

def generate_time_mask(length, gap_length, min_hole, max_hole, context):
    inpainting_indices_time = []
    mask = np.ones(length)
    usable_indices = np.ones(length)
    usable_indices[:context] = 0
    all_holes_length = 0
    iterations = 0
    while all_holes_length < gap_length:
        if iterations > 1000:
            single_hole_len = int((max_hole + min_hole)/2)
            n_holes = int(gap_length / (single_hole_len))
            index = int(length / (n_holes))
            mask = np.ones(length)
            for i in range (n_holes):
                mask[index*(i+1):index*(i+1) + single_hole_len] = 0
                inpainting_indices_time.append((index*(i+1), index*(i+1)+single_hole_len))
            return sorted(inpainting_indices_time, key=lambda idx: idx[0]), mask
        index = random.randrange(length)
        len = random.randrange(min_hole, max_hole)
        if all_holes_length + len > gap_length:
            len = gap_length - all_holes_length
        if usable_indices[index:index+len+context].all() == 1:
            mask[index:index+len] = 0
            usable_indices[index:index+len+context] = 0
            all_holes_length += len
            inpainting_indices_time.append((index, index+len))
        else:
            iterations += 1
            continue
        iterations += 1
    return sorted(inpainting_indices_time, key=lambda idx: idx[0]), mask

multispec_loss_n_fft = (2048, 1024, 512)
multispec_loss_hop_length = (240, 120, 50)
multispec_loss_window_size = (1200, 600, 240)

def multiband_spectral_loss(input_spec, target_spec, slices):
    losses = []
    loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
    for i in range(0, slices):
        losses.append(loss(input_spec[i*input_spec.shape[0]/slices:input_spec.shape[0]+i*input_spec.shape[0]/slices,:], target_spec[i*target_spec.shape[0]/slices:target_spec.shape[0]+i*target_spec.shape[0]/slices,:]))


def compute_loss(device, input, target, alpha, beta, gamma, delta):
    input_spectrogram = input[0,0,:,:] + 1j*input[0,1,:,:]
    target_spectrogram = target[0,0,:,:] + 1j*target[0,1,:,:]

    # input_spectrogram = input + 1j*input
    # target_spectrogram = target + 1j*target

    input_audio = torch.istft(input_spectrogram, n_fft=2047)
    target_audio = torch.istft(target_spectrogram, n_fft=2047)

    loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
    multires_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sc=alpha,
        w_log_mag=beta,
        w_lin_mag=gamma,
        w_phs=delta,
        sample_rate=None,
        scale=None,
        n_bins=None,
        scale_invariance=False,
        device=device
    )

    # input_stft = torch.stft(input, n_fft=2048, return_complex=True)
    # input_channel_a = input_stft.real
    # input_channel_b = input_stft.imag
    #
    # target_stft = torch.stft(target, n_fft=2048, return_complex=True)
    # target_channel_a = target_stft.real
    # target_channel_b = target_stft.imag

    # return loss(torch.stack((input_channel_a, input_channel_b)), torch.stack((target_channel_a, target_channel_b))) + 1/10*multires_loss(input, target)
    # return loss(torch.stack((input_channel_a, input_channel_b)), torch.stack((target_channel_a, target_channel_b))) + 1/10*multires_loss(input, target)
    # return 1/10*multires_loss(input, target)

    return loss(input, target) + 1/10*multires_loss(input_audio, target_audio)
    #return (torch.linalg.norm(target - input)**2/(torch.linalg.norm(target)**2) + 1/5) + 1/5*multires_loss(input_audio, target_audio)
