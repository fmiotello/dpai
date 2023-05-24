### IMPORTS and SETUP ###

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from random import seed
from random import random, randint
import os
from pathlib import Path
import soundfile as sf
import math
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sklearn
import argparse
from random import seed
from random import random, randint
#from __future__ import print_function

#from models.resnet import ResNet
#from models.unet import UNet
from models.skip import skip

import torch
import torch.optim

from sklearn.preprocessing import MinMaxScaler

from utils.inpainting_utils import *
import wandb
from pesq import pesq
from skimage.metrics import structural_similarity as ssim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

#SAMPLE_RATE = 16e3
dim_div_by = 32
files = []
aggregation_type = 'stack' # TODO
inpainting_type = 'short'

### MAIN ###

if __name__ == '__main__':

    # args parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", "--source", type=Path, default=None)
    parser.add_argument("-t", "--type", type=str, default='real-imag')
    parser.add_argument("-c", "--context", type=int, default=None)
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-n", "--normalize", action="store_true")
    parser.add_argument("-i", "--inpainting_type", type=str, default=None)
    args = parser.parse_args()

    if args.source is not None:
        if os.path.isdir(args.source):
            # TODO dir of samples case (calculate accumulated statistics)
            for f in sorted(os.listdir(args.source)):
                files.append(os.path.join(args.source, f))
        else:
            files.append(os.path.abspath(args.source))

    if args.type is not None:
        input_data = args.type  #'mag-phase' or 'real-imag'
        if input_data == 'mag-phase':
            feature_range = (0,100)
        elif input_data == 'real-imag':
            feature_range = (-100,100)
        elif input_data == 'mag-ifreq':
            feature_range = (0,100)

    if args.context is not None:
        context = args.context
    else:
        context = 20

    if args.epochs is not None:
        num_iter = args.epochs
    else:
        num_iter = 5001

    if args.learning_rate is not None:
        LR = args.learning_rate
    else:
        LR = 0.001

    if args.inpainting_type is not None:
        inpainting_type = args.inpainting_type


    results_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/results/' + os.path.basename(files[0].split('.')[0])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.normalize:
        norm = 'norm-' + str(feature_range[1])
    else:
        norm = 'no-norm'

    hyperparameter_defaults = {
        'filter_size': 3,
        'loss': 'mse',
        'noise_method': 'noise',
        'reg_noise' : True,
        'num_channels': [128] * 5,
        'num_channels_skip': [0],
        'downsample_mode': 'nearest'
    }

    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    dir_path = results_path + '/'  + datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + inpainting_type + '_ksize-' + str(config['filter_size']) + '_loss-' + config['loss'] + '_' + config['noise_method'] + '_' + config['downsample_mode'] + '_regnoise-' + str(config['reg_noise']) + '_channels-[' + str(config['num_channels']) + ']_skip-[' + str(config['num_channels_skip']) + ']'
    dirpath = dir_path.replace(' ', '-')
    os.makedirs(dir_path)


    for file in enumerate(files): # BIG CYCLE OF ALL DATASET FILES

        print('\n[PROCESSING sample: ' + os.path.basename(file[1]) + '] %d/%d' %(file[0]+1, len(files)))

        startTime = datetime.now()

        # SPECTROGRAM
        source_audio, sr = librosa.load(file[1]) #, sr=SAMPLE_RATE)
        #source_audio = source_audio[:1024*256] #mod
        X = librosa.stft(source_audio) #, n_fft=1024, hop_length=256) #mod
        X_crop = crop_spectrogram(X, dim_div_by)

        X_a, X_b = separate_spectrogram(X_crop, input_data, sr)

        if args.normalize:
            scaler_a = MinMaxScaler(feature_range=feature_range)
            scaler_b = MinMaxScaler(feature_range=feature_range)

            X_a_norm = scaler_a.fit_transform(X_a)
            X_b_norm = scaler_b.fit_transform(X_b)

            input_np = aggregate(X_a_norm, X_b_norm, aggregation_type)
        else:
            input_np = aggregate(X_a, X_b, aggregation_type) # TODO add aggregation type as command line argument

        # X_mag, X_phase = librosa.magphase(X_crop)
        # X_mag_max = np.max(X_mag)
        # X_mag = X_mag/X_mag_max
        # X_norm = X_mag * X_phase # normalized spectrogram

        #X_a_norm, X_b_norm = separate_spectrogram(X_norm, input_data)

        mask = generate_mask(X_a.shape, inpainting_type) #context, s = 20, min_length = 1, max_length = 1, probability = 0.01)
        #mask = np.concatenate((np.ones((512, 480)), np.zeros((512, 64)), np.ones((512,480))), axis = 1)

        input_mask_np = aggregate(mask, mask, aggregation_type) # TODO change variable names input_np, input_mask_np... they are misleading

        plt.figure(figsize=(14,10))
        librosa.display.specshow(input_mask_np[0], sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(dir_path + '/mask.png')


        #NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
        pad = 'reflection' # 'zero'
        OPT_OVER = 'net'
        OPTIMIZER = 'adam'

        INPUT = config['noise_method']
        input_depth = 16
        param_noise = False
        #show_every = 5
        #figsize = 5
        if config['reg_noise']:
            reg_noise_std = 0.03
        else:
            reg_noise_std = 0

        net = skip(input_depth, input_np.shape[0],
                   num_channels_down = config['num_channels'],
                   num_channels_up   = config['num_channels'],
                   num_channels_skip = config['num_channels_skip'],
                   upsample_mode='nearest', filter_skip_size=1, filter_size_up=config['filter_size'], filter_size_down=config['filter_size'],
                   need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        net = net.type(dtype)

        net_input = get_noise(input_depth, INPUT, (np.shape(input_np)[1], np.shape(input_np)[2])).type(dtype)
        input_tensor = torch.from_numpy(input_np)[None,  :].type(dtype) #img_var = np_to_torch(spec_np).type(dtype)
        input_mask_tensor = torch.from_numpy(input_mask_np)[None, :].type(dtype) #mask_var = np_to_torch(spec_mask_np).type(dtype)


        # Compute number of parameters
        s  = sum(np.prod(list(p.size())) for p in net.parameters())
        #print ('Number of params: %d' % s)

        # Loss
        if config['loss'] == 'mse':
            loss = torch.nn.MSELoss().type(dtype)
        elif config['loss'] == 'msa':
            loss = torch.nn.L1Loss().type(dtype)

        # FITTING CYCLE

        i = 0
        #specs = []
        loss_values = np.zeros(num_iter)
        validation_loss_values = np.zeros(num_iter)
        inverted_mask_tensor = 1 - input_mask_tensor
        #epochs = 0

        wandb_metrics = {   "loss": loss_values[i],
                            "validation_loss": validation_loss_values[i]   }

        def closure():

            global i

            if param_noise:
                for n in [x for x in net.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std() / 50

            net_input = net_input_saved
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

            out = net(net_input)


            total_loss = loss(out * input_mask_tensor, input_tensor * input_mask_tensor)
            total_validation_loss = loss(out * inverted_mask_tensor, input_tensor * inverted_mask_tensor)

            loss_values[i] = total_loss.item()
            validation_loss_values[i] = total_validation_loss.item()

            wandb_metrics['loss'] = loss_values[i]
            wandb_metrics['validation_loss'] = validation_loss_values[i]

            wandb.log(wandb_metrics)

            total_loss.backward()

            if i%10 == 0: #prints loss every 10 iterations
                print('Iteration %05d/%05d:    Loss %f | Validation loss %f' % (i, num_iter, total_loss.item(), total_validation_loss.item()))#, '\r', end='')
            #if  i%show_every == 0:
            #    out_np = torch_to_np(out)
            #    plt.imshow(out_np[0,:,:], aspect='auto', origin='lower')
            #    plt.show()
            #epochs = i
            i += 1

            return total_loss

        net_input_saved = net_input.detach().clone()
        noise =  net_input.detach().clone()

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

        elapsed_time = datetime.now() - startTime

        inner_path = dir_path + '/' + os.path.basename(file[1]).split('.')[0]
        os.makedirs(inner_path)

        plt.figure(figsize = (14, 10))
        plt.plot(loss_values)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(inner_path + '/loss.png')

        plt.figure(figsize = (14, 10))
        plt.plot(validation_loss_values)
        plt.xlabel('epochs')
        plt.ylabel('validation_loss')
        plt.savefig(inner_path + '/validation_loss.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_a * mask), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/original_masked.png')

        out_np = torch_to_np(net(net_input))

        original_spectrogram = reconstruct_spectrogram(X_a, X_b, input_data, sr)
        original_audio = librosa.istft(original_spectrogram)
        original_audio_masked = librosa.istft(reconstruct_spectrogram(X_a*mask, X_b*mask, input_data, sr))

        X_out_a, X_out_b = disaggregate(out_np, aggregation_type) # TODO

        # X_out_n = X_out_a + 1j*X_out_b # TODO
        # X_mag_n, X_phase_n = librosa.magphase(X_out_n)
        # X_out_denorm = X_mag_n*X_mag_max * X_phase_n
        #
        # X_out_a, X_out_b = separate_spectrogram(X_out_denorm, input_data)

        if args.normalize:
            X_out_a = scaler_a.inverse_transform(X_out_a)
            X_out_b = scaler_b.inverse_transform(X_out_b)

        reconstructed_spectrogram = reconstruct_spectrogram(X_out_a, X_out_b, input_data, sr)
        reconstructed_audio = librosa.istft(reconstructed_spectrogram)

        #nmse_a = nmse(X_a, X_out_a)
        #nmse_b = nmse(X_b, X_out_b)
        nmse_spectrogram = nmse(original_spectrogram, reconstructed_spectrogram)
        nmse_audio = nmse(original_audio, reconstructed_audio)

        inverted_mask = 1-mask
        if inverted_mask.any():
            original_audio_masked_part = librosa.istft(reconstruct_spectrogram(X_a*inverted_mask, X_b*inverted_mask, input_data, sr))
            reconstructed_audio_masked_part = librosa.istft(reconstruct_spectrogram(X_out_a*inverted_mask, X_out_b*inverted_mask, input_data, sr))
            nmse_audio_masked_part = nmse(original_audio_masked_part, reconstructed_audio_masked_part)
        else:
            nmse_audio_masked_part = None

        save_plots(inner_path, X_a, X_b, X_out_a, X_out_b, input_data, sr)

        sf.write(inner_path + '/original_audio.wav', original_audio, sr, 'PCM_24')
        sf.write(inner_path + '/original_audio_masked.wav', original_audio_masked, sr, 'PCM_24')
        sf.write(inner_path + '/reconstructed_audio.wav', reconstructed_audio, sr, 'PCM_24')

        metrics = {
            #'epochs': epochs,
            # 'learning_rate': LR,
            # 'input_data': input_data,
            # 'normalization': None,
            # 'aggregation_type': 'stack',
            'lowest_training_loss': np.min(loss_values),
            'last_training_loss': loss_values[-1],
            'lowest_validation_loss': np.min(validation_loss_values),
            'last_validation_loss': validation_loss_values[-1],
            #'nmse_' + input_data.split('-')[0]: nmse_a,
            #'nmse_' + input_data.split('-')[1]: nmse_b,
            'nmse_spectrogram': nmse_spectrogram,
            'nmse_audio': nmse_audio,
            'nmse_audio_masked_part': nmse_audio_masked_part,
            'pesq': pesq(16e3, np.array(librosa.resample(original_audio, sr, 16e3)), np.array(librosa.resample(reconstructed_audio, sr, 16e3)), 'wb'),
            'ssim': ssim(X_a, X_out_a), #only real part
            'elapsed_time': str(elapsed_time).split('.')[0],
            'loss': loss_values.tolist(),
            'validation_loss': validation_loss_values.tolist()
        }

        if args.normalize:
            metrics['normalization'] = feature_range

        with open(inner_path + '/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        # metrics_file = open(inner_path + '/metrics.txt', "w")
        #
        # metrics_file.write('===METRICS===\n')
        # metrics_file.write('\nEpochs: ' + str(num_iter))
        # metrics_file.write('\nLearning rate: ' + str(LR))
        # metrics_file.write('\nInput data type: ' + input_data)
        # if args.normalize:
        #     metrics_file.write('\nNormalization: ' + str(feature_range))
        # else:
        #     metrics_file.write('\nNormalization: None')
        # metrics_file.write('\nAggregation type: ' 'stack') # TODO
        # metrics_file.write('\n\nLowest training loss: ' + str(np.min(loss)))
        # metrics_file.write('\nLast training loss: ' + str(loss[-1]))
        # metrics_file.write('\n\nLowest validation loss: ' + str(np.min(validation_loss)))
        # metrics_file.write('\nLast validation loss: ' + str(validation_loss[-1]))
        #
        # metrics_file.write('\n\nNMSE ' + input_data.split('-')[0] + ' part: ' + str(nmse_a))
        # metrics_file.write('\nNMSE ' + input_data.split('-')[1] + ' part: ' + str(nmse_b))
        # metrics_file.write('\nNMSE reconstructed spectrogram: ' + str(nmse_spectrogram))
        # metrics_file.write('\nNMSE reconstructed audio: ' + str(nmse_audio))
        #
        # elapsed_time = datetime.now() - startTime
        #
        # metrics_file.write('\n\nExecution time: ' + str(elapsed_time).split('.')[0])
        # metrics_file.close()

        plt.close('all')

        print('\n[DONE]\n')

    '''

    print('\n[COMPUTING GLOBAL METRICS]')
    samples =  [item for item in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, item))]

    #all_nmse_a = []
    #all_nmse_b = []
    all_nmse_spectrogram = []
    all_nmse_audio = []
    all_nmse_audio_masked_part = []
    all_pesq = []
    all_ssim = []

    for sample in samples:
        dir = os.path.join(dir_path, sample)
        f = open(dir + '/metrics.json')
        metrics = json.load(f)

        #all_nmse_a.append(metrics['nmse_' + input_data.split('-')[0]])
        #all_nmse_b.append(metrics['nmse_' + input_data.split('-')[1]])
        all_nmse_spectrogram.append(metrics['nmse_spectrogram'])
        all_nmse_audio.append(metrics['nmse_audio'])
        all_nmse_audio_masked_part.append(metrics['nmse_audio_masked_part'])
        all_pesq.append(metrics['pesq'])
        all_ssim.append(metrics['ssim'])

    dataset_metrics = {
        'epochs': num_iter,
        'learning_rate': LR,
        'input_data': input_data,
        'normalization': None,
        'inpainting_type': inpainting_type,
        #'mean_nmse_' + input_data.split('-')[0]: np.mean(all_nmse_a),
        #'mean_nmse_' + input_data.split('-')[1]: np.mean(all_nmse_b),
        'mean_nmse_spectrogram': np.mean(all_nmse_spectrogram),
        'mean_nmse_audio': np.mean(all_nmse_audio),
        'var_nmse_audio': np.var(all_nmse_audio),
        'mean_nmse_audio_masked_part': None,
        'var_nmse_audio_masked_part': None,
        'mean_pesq': np.mean(all_pesq),
        'var_pesq': np.var(all_pesq),
        'mean_ssim': np.mean(all_ssim),
        'var_ssim': np.var(all_ssim),
    }

    if all(el != None for el in all_nmse_audio_masked_part):
        dataset_metrics['mean_nmse_audio_masked_part'] = np.mean(all_nmse_audio_masked_part)
        dataset_metrics['var_nmse_audio_masked_part'] = np.var(all_nmse_audio_masked_part)

    with open(dir_path + '/dataset_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_metrics, f, ensure_ascii=False, indent=4)

    '''

    print('\n[COMPLETED]\n')
