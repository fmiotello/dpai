import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from .common_utils import *
from random import seed
from random import random, randint
import math
import matplotlib.pyplot as plt
import librosa

def get_text_mask(for_image, sz=20):
    font_fname = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    font_size = sz
    font = ImageFont.truetype(font_fname, font_size)

    img_mask = Image.fromarray(np.array(for_image)*0+255)
    draw = ImageDraw.Draw(img_mask)
    draw.text((128, 128), "hello world", font=font, fill='rgb(0, 0, 0)')

    return img_mask

def get_bernoulli_mask(for_image, zero_fraction=0.95):
    img_mask_np=(np.random.random_sample(size=pil_to_np(for_image).shape) > zero_fraction).astype(int)
    img_mask = np_to_pil(img_mask_np)

    return img_mask

#def generate_mask(input_dimensions, context, s = 30, min_length = 1, max_length = 50, probability = 0.5): #context = how much data before and after a hole, masking_type = vertical | total
    # mask = np.ones(input_dimensions)
    # len = 0
    # seed(s)
    # i = context
    # j = context
    # while (i < input_dimensions[1]-context):
    #     if (random() < probability):
    #         len = randint(min_length, max_length)
    #         for j in range(0, len):
    #             mask[:,i+j] = 0
    #         i += len + context
    #         continue
    #     i += 1
    # return mask

def generate_mask(input_dimensions, type): #, context):

    # at sr=22050 n_fft=2048 hop_length=n_fft/4 1 time frame corresponds to more or less 20ms

    if type == 'short':
        len = 3
        context = 10
    elif type == 'mid':
        len = 6
        context = 60
    elif type == 'long':
        len = 10
        context = 90

    mask = np.ones(input_dimensions)
    i = context
    j = context
    while (i < input_dimensions[1]-context):
        for j in range(0, len):
            mask[:,i+j] = 0
        i += len + context
    return mask

def separate_spectrogram(spectrogram, input_type, sr):
    if input_type == 'mag-phase':
        return np.abs(spectrogram), np.angle(spectrogram)
    elif input_type == 'real-imag':
        return spectrogram.real, spectrogram.imag
    elif input_type == 'mag-ifreq':
        ifreq = np.diff(np.unwrap(np.angle(spectrogram)))
        ifreq = np.column_stack((np.angle(spectrogram)[:,0], ifreq))
        return np.abs(spectrogram), ifreq

def crop_spectrogram(spectrogram, dim_div_by):
    return spectrogram[0:(spectrogram.shape[0]//dim_div_by)*dim_div_by,0:(spectrogram.shape[1]//dim_div_by)*dim_div_by]

def aggregate(X_a, X_b, aggregation):
    if aggregation == 'stack':
        return np.stack((X_a, X_b))
    elif aggregation == 'concat':
        return np.concat((X_a, X_b)) # TODO

def disaggregate(X, aggregation):
    if aggregation == 'stack':
        return X[0], X[1]
    elif aggregation == 'concat':
        return X[0], X[1] # TODO

def reconstruct_spectrogram(X_a, X_b, input_type, sr):
    if input_type == 'mag-phase':
        return X_a * np.exp(1j*X_b)
    elif input_type == 'real-imag':
        return X_a + 1j*X_b
    elif input_type == 'mag-ifreq':
        return X_a * np.exp(1j*np.cumsum(X_b, axis=1))

def nmse(original, reconstructed):
    return 10*math.log10(np.linalg.norm(reconstructed - original)**2/(np.linalg.norm(original)**2))

# def scale(X, min, max):
#     return min + ((X - np.min(X))*(max - min))/(np.max(X) - np.min(X))

def save_plots(path, original_a, original_b, reconstructed_a, reconstructed_b, input_type, sample_rate):
    if input_type == 'mag-phase':
        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_mag.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(original_b, sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_phase.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_mag.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(reconstructed_b, sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_phase.png')

    elif input_type == 'real-imag':
        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_real.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_b), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_imag.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_real.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_b), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_imag.png')

    elif input_type == 'mag-ifreq':
        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_mag.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(original_b, sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/original_ifreq.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_a), sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_mag.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(reconstructed_b, sr=sample_rate, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(path + '/reconstructed_ifreq.png')

    plt.close('all')
