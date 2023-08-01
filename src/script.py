from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import os
from pathlib import Path
import soundfile as sf
import json
import sklearn
import argparse
from models.mulresunet import MulResUnet
#from models.mulresunet import MulResUnet3D
from models.unet import UNet
import torch
import torch.optim
from sklearn.preprocessing import MinMaxScaler
from utils.common_utils import *
import wandb
from pesq import pesq
from skimage.metrics import structural_similarity as ssim
from torchinfo import summary
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
dim_div_by = 2



def main(args, device):

    files = []

    if not os.path.isdir(args.source):
        files.append(os.path.abspath(args.source))
    else:
        files = extract_from_dataset(args.source, args.n_audio)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    f = open(args.net)
    net_specs = json.load(f)

    hyperparameters = {
        'net_type': net_specs['name'],
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'activation_function': args.act_fun,
        'input_depth': args.input_depth,
        'input_noise': args.input_noise,
        'noise': args.noise,
        'reg_noise' : True,
    }

    dir_path = args.results_dir + '/'  + datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = dir_path.replace(', ', '-')
    os.makedirs(dir_path)

    with open(dir_path + '/net_specs.json', 'w', encoding='utf-8') as f:
        json.dump(net_specs, f, ensure_ascii=False, indent=4)

    for file in files: # BIG CYCLE OF ALL DATASET FILES

        inner_path = dir_path + '/' + os.path.basename(file).split('.')[0]
        os.makedirs(inner_path)
        if args.output_every != 0:
            os.makedirs(inner_path + '/intermediate_outputs/imag')
            os.makedirs(inner_path + '/intermediate_outputs/audio')

        print('\n[PROCESSING sample: ' + os.path.basename(file) + ']')

        start_time = datetime.now()

        source_audio, sr = librosa.load(file, sr=16000, offset=args.offset, duration=args.audio_len)

        X = librosa.stft(source_audio)
        X = X[:1024,:208]

        #X = crop_spectrogram(X, dim_div_by)

        X_a, X_b = X.real, X.imag

        #source_audio = source_audio[:librosa.istft(X).shape[0]]

        ref_np = np.stack((X_a, X_b))
        ref_tensor = torch.from_numpy(ref_np)[None,  :].type(dtype).to(device)

        # tot_gap = int(args.tot_gap*1e-3*sr)
        # min_gap = int(args.min_gap*1e-3*sr)
        # max_gap = int(args.max_gap*1e-3*sr)
        # context = int(args.context*1e-3*sr)

        # for date in os.listdir(os.path.join('/nas', 'home', 'fmiotello', 'inpainting', 'deep-prior-audio-inpainting', 'results', 'FINAL_RESULTS', 'gap'+str(args.tot_gap), args.results_dir.split('/')[-1])):
        #     if not os.path.isdir(os.path.join('/nas', 'home', 'fmiotello', 'inpainting', 'deep-prior-audio-inpainting', 'results', 'FINAL_RESULTS', 'gap'+str(args.tot_gap), args.results_dir.split('/')[-1], date)):
        #         continue
        #     f = open(os.path.join('/nas', 'home', 'fmiotello', 'inpainting', 'deep-prior-audio-inpainting', 'results', 'FINAL_RESULTS', 'gap'+str(args.tot_gap), args.results_dir.split('/')[-1], date, os.path.basename(file).split('.')[0], 'metrics.json'))
        #     holes = json.load(f)

        # inpainting_indices_time = holes['inpainting_indices_time']#generate_time_mask(source_audio.shape[0], tot_gap, min_gap, max_gap, context)
        # inpainting_indices_time, mask_time = generate_time_mask(source_audio.shape[0], tot_gap, min_gap, max_gap, context)
        # mask_time = np.ones(source_audio.shape[0])
        # for hole in inpainting_indices_time:
        #     mask_time[hole[0]:hole[1]] = 0

        # mask_freq = librosa.stft(mask_time)
        # mask = np.tile(mask_freq[0,:], (mask_freq.shape[0], 1))
        # mask = np.abs(mask)/np.max(np.abs(mask))

        #mask = generate_mask(X_a.shape, 5, 5)

        # mask = np.where(mask != 1, 0, 1)

        # inpainting_indices_freq = []
        # i = 0
        # while i < len(mask[0,:]):
        #     if (mask[:,i] != 1).all():
        #         j = 0
        #         while i+j < len(mask[0,:]) and (mask[:,i:i+j] != 1).all():
        #             j += 1
        #         inpainting_indices_freq.append((i, i+j))
        #         i += j
        #     else:
        #         i += 1
        #
        # tot_hole_time = 0
        # hop_size = 512
        # time_per_frame = hop_size/sr  # hop_length / sample_rate
        # for idx in inpainting_indices_freq:
        #     tot_hole_time += (idx[1] - idx[0])*time_per_frame
        #
        # inpainting_indices_freq_new = inpainting_indices_freq
        # inpainting_indices_time_new = inpainting_indices_time


        # while tot_hole_time*1e3 > args.tot_gap + 20:
        #     hole = inpainting_indices_freq[np.random.choice(len(inpainting_indices_freq))]
        #     # idx = inpainting_indices_freq.index(hole)
        #     if hole[0] == hole[1] - 1:
        #         continue
        #     mask[:, hole[1]-1:hole[1]] = 1
        #     inpainting_indices_freq_new.remove(hole)
        #     # inpainting_indices_time_new[idx][1] -= 250
        #     # if (inpainting_indices_time_new[idx][1] < inpainting_indices_time_new[idx][0]):
        #     #     inpainting_indices_time_new.del(idx)
        #     if (hole[0] != hole[1] - 1):
        #         inpainting_indices_freq_new.append((hole[0], hole[1]-1))
        #     tot_hole_time -= time_per_frame
        #
        # inpainting_indices_freq_new = sorted(inpainting_indices_freq_new, key=lambda idx: idx[0])

        # inpainting_indices_time_new = []
        # for hole in inpainting_indices_freq:
        #     start = hole[0]*time_per_frame*sr
        #     end = hole[1]*time_per_frame*sr
        #     inpainting_indices_time_new.append((int(start), int(end)))

        # rms = librosa.feature.rms(S=np.abs(X))

        #mask = np.where(rms < 0.01, 0, 1)
        #mask = np.where(np.logical_and(rms < 0.01, rms > 0.0028), 0, 1) # for fdenoised sample
        #mask = np.where(np.logical_and(rms < 0.025, rms > 0.018), 0, 1)
        # mask = np.where(rms < 0.02, 0, 1) # denoised
        #mask = np.where(rms < 0.035, 0, 1) # noised

        # i = 0
        # j = 0
        #
        # while i < X.shape[1]:
        #     while i+j < X.shape[1] and (mask[:,i:i+j] == 0).all():
        #         j += 1
        #     if j > 10:
        #         mask[:,i:i+j-1] = 1
        #         i += j
        #         j = 0
        #     else:
        #         if i == j:
        #             i += 1
        #         else:
        #             i += j
        #             j = 0

        # mask = np.tile(mask, (X.shape[0], 1))

        mask = np.ones(X_a.shape)
        mask[:,20:25] = 0
        mask[:,50:55] = 0
        mask[:,100:105] = 0
        mask[:,120:125] = 0
        mask[:,150:155] = 0

        mask_np = np.stack((mask, mask))
        mask_tensor = torch.from_numpy(mask_np)[None, :].type(dtype).to(device)

        # plt.figure(figsize=(14,10))
        # librosa.display.specshow(mask_np[0], sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        # plt.colorbar()
        # plt.savefig(inner_path + '/mask.png')

        with open(inner_path + '/mask.npy', 'wb') as f:
            np.save(f, mask)

        OPT_OVER = 'net'
        OPTIMIZER = 'adam'

        INPUT = 'noise'
        param_noise = False

        if hyperparameters['reg_noise']:
            reg_noise_std = 0.03
        else:
            reg_noise_std = 0

        # net = UNet(
        #     num_input_channels=hyperparameters['input_depth'],
        #     num_output_channels=ref_np.shape[0],
        #     need_sigmoid=False,
        #     feature_scale=4,
        #     more_layers=0,
        #     concat_x=False,
        #     upsample_mode='nearest'
        # ).type(dtype).to(device)

        net = MulResUnet(
                num_input_channels=hyperparameters['input_depth'],
                num_output_channels=ref_np.shape[0],
                num_channels_down=net_specs['num_channels_down'],
                num_channels_up=net_specs['num_channels_up'],
                num_channels_skip=net_specs['num_channels_skip'],
                kernel_size_down=net_specs['kernel_size_down'],
                kernel_size_up=net_specs['kernel_size_up'],
                kernel_size_skip=net_specs['kernel_size_skip'],
                kernel_size_downsampling=net_specs['kernel_size_downsampling'],
                conv_type_down=net_specs['conv_type_down'],
                conv_type_up=net_specs['conv_type_up'],
                conv_type_res=net_specs['conv_type_res'],
                conv_type_downsampling=net_specs['conv_type_downsampling'],
                conv_type_out=net_specs['conv_type_out'],
                anchor_down=net_specs['anchor_down'],
                anchor_up=net_specs['anchor_up'],
                anchor_res=net_specs['anchor_res'],
                anchor_downsampling=net_specs['anchor_downsampling'],
                anchor_out=net_specs['anchor_out'],
                alpha=1,
                last_act_fun=None,
                need_bias=True,
                upsample_mode=net_specs['upsample_mode'],
                act_fun='LeakyReLU',
                dropout=net_specs['dropout']
        ).type(dtype).to(device)

        net_input = get_noise(hyperparameters['input_depth'], INPUT, (np.shape(ref_np)[1], np.shape(ref_np)[2]), noise_type=hyperparameters['input_noise']).type(dtype).to(device)

        # Compute number of parameters
        s  = sum(np.prod(list(p.size())) for p in net.parameters())
        print('Number of params: %d' % s)

        # print(net)


        # FITTING CYCLE

        current_epoch = 0
        loss_values = np.zeros(hyperparameters['epochs'])

        validation_loss_values = np.zeros(hyperparameters['epochs'])
        inverted_mask_tensor = (1 - mask_tensor).to(device)

        # best_validation_loss = 1e8
        # best_epoch = 0
        # out_best = None


        # wandb_metrics = {
        #     'loss': 0,
        #     'validation_loss': 0
        # }

        # mask_time_tensor = torch.tensor(mask_time).to(device)
        # source_audio_tensor = torch.tensor(source_audio).to(device)


        def closure():

            nonlocal current_epoch
            # nonlocal best_validation_loss
            # nonlocal best_epoch
            # nonlocal out_best

            if param_noise:
                for n in [x for x in net.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std() / 50

            net_input = net_input_saved
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

            out = net(net_input)


            # loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
            # loss1 = loss(out * mask_tensor, ref_tensor * mask_tensor)
            # validation_loss1 = loss(out * inverted_mask_tensor, ref_tensor * inverted_mask_tensor)

            # out_time = torch.istft(out[0,0,:,:] + 1j*out[0,1,:,:], n_fft=2048)
            # limit = out_time.shape[0]

            total_loss = compute_loss(device, out * mask_tensor, ref_tensor * mask_tensor, args.alpha, args.beta, args.gamma, args.delta)
            total_validation_loss = compute_loss(device, out * inverted_mask_tensor, ref_tensor * inverted_mask_tensor, args.alpha, args.beta, args.gamma, args.delta)

            # total_loss = compute_loss(device, out_time * mask_time_tensor[0:limit], source_audio_tensor[0:limit] * mask_time_tensor[0:limit])
            # total_validation_loss = compute_loss(device, out_time * (1 - mask_time_tensor[0:limit]), source_audio_tensor[0:limit] * (1 - mask_time_tensor[0:limit]))

            # total_loss = loss1 + loss2
            # total_validation_loss = validation_loss1 + validation_loss2

            loss_values[current_epoch] = total_loss.item()
            validation_loss_values[current_epoch] = total_validation_loss.item()

            # wandb_metrics['loss'] = loss_values[current_epoch]
            # wandb_metrics['validation_loss'] = validation_loss_values[current_epoch]
            # wandb.log(wandb_metrics)

            total_loss.backward()



            if current_epoch%10 == 0: #prints loss every 10 iterations
                print('[' + os.path.basename(file) + '] ' + 'Iteration %05d/%05d:    Loss %f | Validation loss %f' % (current_epoch, hyperparameters['epochs'], total_loss.item(), total_validation_loss.item()))

            # if total_validation_loss.item() < best_validation_loss:
            #     best_validation_loss = total_validation_loss.item()
            #     best_epoch = current_epoch
            #     out_best = out

            if args.output_every != 0 and current_epoch%args.output_every == 0:
                out_np = out.detach().cpu().numpy()[0]
                X_out_a, X_out_b = out_np[0], out_np[1]

                plt.figure(figsize=(14,10))
                librosa.display.specshow(librosa.amplitude_to_db(X_out_a + 1j*X_out_b), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
                plt.colorbar()
                plt.savefig(inner_path + '/intermediate_outputs/imag/epoch%05d_magnitude.png' % (current_epoch))
                plt.close('all')

                reconstructed_spectrogram = X_out_a + 1j*X_out_b
                reconstructed_audio = librosa.istft(reconstructed_spectrogram)

                sf.write(inner_path + '/intermediate_outputs/audio/epoch%05d_audio.wav' % (current_epoch), reconstructed_audio, sr, 'PCM_24')

            current_epoch += 1

            return total_loss

        net_input_saved = net_input.detach().clone()
        noise =  net_input.detach().clone()

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, hyperparameters['learning_rate'], hyperparameters['epochs'])

        elapsed_time = datetime.now() - start_time

        ### SAVE BEST EPOCH AUDIO AND SPEC ###

        # out_np_best = out_best.detach().cpu().numpy()[0]
        # X_out_a_best, X_out_b_best = out_np_best[0], out_np_best[1]
        #
        # plt.figure(figsize=(14,10))
        # librosa.display.specshow(librosa.amplitude_to_db(X_out_a_best + 1j*X_out_b_best), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        # plt.colorbar()
        # plt.savefig(inner_path + '/epoch%05d_best_magnitude.png' % (best_epoch))
        # plt.close('all')
        #
        # reconstructed_spectrogram = X_out_a_best + 1j*X_out_b_best
        # reconstructed_audio = librosa.istft(reconstructed_spectrogram)
        #
        # sf.write(inner_path + '/epoch%05d_best_audio.wav' % (best_epoch), reconstructed_audio, sr, 'PCM_24')

        ######################################


        with open(inner_path + '/hparams.json', 'w', encoding='utf-8') as f:
            json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

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

        net_input_np = net_input.detach().cpu().numpy()[0]
        net_input_np_a, net_input_np_b = net_input_np[0], net_input_np[1]

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(net_input_np_a + 1j*net_input_np_b), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/net_input.png')

        # if args.layers:
        #
        #     return_layers = {}
        #
        #     for name, module in net.named_modules():
        #         if not ('dr' in name or 'net' in name or name=='' or 'zoom' in name):
        #             return_layers[name] = name
        #             #print(name, module)
        #
        #
        #     mid_getter = MidGetter(net, return_layers=return_layers, keep_output=True)
        #     mid_outputs, out_np = mid_getter(net_input)
        #
        #     layers_path = inner_path + '/mid_layers'
        #     os.makedirs(layers_path)
        #
        #     for key in return_layers:
        #         layer = mid_outputs[key].detach().cpu().numpy()[0]
        #         mean_layer = np.mean(layer, axis=0)
        #
        #         plt.figure(figsize=(14,10))
        #         librosa.display.specshow(librosa.amplitude_to_db(mean_layer), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        #         plt.colorbar()
        #         plt.savefig(layers_path + '/' + return_layers[key] + '.png')
        #
        #         plt.close('all')
        #
        #     out_np = out_np.detach().cpu().numpy()[0]
        #
        # else:

        out_np = net(net_input).detach().cpu().numpy()[0]

        original_spectrogram = X_a + 1j*X_b
        original_audio = librosa.istft(original_spectrogram)
        original_audio_masked = librosa.istft(X_a*mask + 1j*X_b*mask)

        X_out_a, X_out_b = out_np[0], out_np[1]

        reconstructed_spectrogram = X_out_a + 1j*X_out_b
        reconstructed_audio = librosa.istft(reconstructed_spectrogram)

        nmse_spectrogram = nmse(original_spectrogram, reconstructed_spectrogram)
        nmse_audio = nmse(original_audio, reconstructed_audio)

        inverted_mask = 1-mask

        original_audio_masked_part = librosa.istft(X_a*inverted_mask + 1j*X_b*inverted_mask)
        reconstructed_audio_masked_part = librosa.istft(X_out_a*inverted_mask + 1j*X_out_b*inverted_mask)
        nmse_audio_masked_part = nmse(original_audio_masked_part, reconstructed_audio_masked_part)

        reconstructed_spectrogram_with_original_context = (X_a*mask + X_out_a*inverted_mask) + 1j*(X_b*mask + X_out_b*inverted_mask)

        nmse_audio_with_original_context = nmse(original_audio, librosa.istft(reconstructed_spectrogram_with_original_context))
        nmse_audio_masked_part_with_original_context = nmse(original_audio, librosa.istft(reconstructed_spectrogram_with_original_context))

        plt.figure(figsize=(14,5))
        librosa.display.waveplot(original_audio, sr=sr)
        plt.savefig(inner_path + '/waveform_original.png')

        plt.figure(figsize=(14,5))
        librosa.display.waveplot(original_audio_masked, sr=sr)
        plt.savefig(inner_path + '/waveform_original_masked.png')

        plt.figure(figsize=(14,5))
        librosa.display.waveplot(reconstructed_audio, sr=sr)
        plt.savefig(inner_path + '/waveform_reconstructed.png')

        original_mag, _ = librosa.magphase(X)

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_mag), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/magnitude_original.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(original_mag*mask), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/magnitude_original_masked.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_spectrogram_with_original_context), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/reconstructed_spectrogram_with_original_context.png')

        reconstructed_mag, _ = librosa.magphase(reconstructed_spectrogram)

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(reconstructed_mag), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/magnitude_reconstructed.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_a), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/real_original.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_b), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/imag_original.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_a * mask), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/real_original_masked.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_b * mask), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/imag_original_masked.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_out_a), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/real_reconstructed.png')

        plt.figure(figsize=(14,10))
        librosa.display.specshow(librosa.amplitude_to_db(X_out_b), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar()
        plt.savefig(inner_path + '/imag_reconstructed.png')

        sf.write(inner_path + '/audio_original.wav', original_audio, sr, 'PCM_24')
        sf.write(inner_path + '/audio_original_masked.wav', original_audio_masked, sr, 'PCM_24')
        sf.write(inner_path + '/audio_reconstructed.wav', reconstructed_audio, sr, 'PCM_24')
        sf.write(inner_path + '/audio_reconstructed_with_original_context.wav', librosa.istft(reconstructed_spectrogram_with_original_context), sr, 'PCM_24')

        try:
            pesq_before = pesq(16e3, original_audio, original_audio_masked, 'wb')
            pesq_after = pesq(16e3, original_audio, reconstructed_audio, 'wb')
        except:
            pesq_before = 999
            pesq_after = 999


        metrics = {
            'file': file,
            'epochs': args.epochs,
            #'gaps_number': args.ngaps,
            #'gaps_length': args.gap_length,
            'lowest_training_loss': np.min(loss_values),
            'last_training_loss': loss_values[-1],
            'lowest_validation_loss': np.min(validation_loss_values),
            'last_validation_loss': validation_loss_values[-1],
            'nmse_spectrogram': nmse_spectrogram,
            'nmse_audio': nmse_audio,
            'nmse_audio_masked_part': nmse_audio_masked_part,
            'nmse_audio_with_original_context': nmse_audio_with_original_context,
            'nmse_audio_masked_part_with_original_context': nmse_audio_masked_part_with_original_context,
            'pesq_before': pesq_before,
            'pesq_after': pesq_after,
            #'ssim_before': ssim(X_a, X_a*mask), #only real part
            #'ssim': ssim(X_a, X_out_a), #only real part
            'elapsed_time': str(elapsed_time).split('.')[0],
            'alpha': args.alpha,
            'beta': args.beta,
            'gamma': args.gamma,
            'delta': args.delta,
            #'audio_len': args.audio_len,
            #'offset': args.offset,
            #'tot_gap': args.tot_gap,
            #'min_gap': args.min_gap,
            #'max_gap': args.max_gap,
            #'context': args.context,
            #'real_tot_gap': tot_hole_time,
            #'inpainting_indices_freq': inpainting_indices_freq,
            #'inpainting_indices_time': inpainting_indices_time,
            #'inpainting_indices_freq_new': inpainting_indices_freq_new,
            #'inpainting_indices_sec': list(map(lambda idx: (idx[0]/2, idx[1]/2), inpainting_indices_time))
        }

        losses = {
            'loss': list(loss_values),
            'validation_loss': list(validation_loss_values),
        }

        with open(inner_path + '/losses.json', 'w', encoding='utf-8') as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)

        with open(inner_path + '/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        ### MODIFIED INPUT ###

        # if args.mod:
        #
        #     new_net_input = get_noise(hyperparameters['input_depth'], INPUT, (np.shape(ref_np)[1], np.shape(ref_np)[2]), noise_type=hyperparameters['input_noise']).type(dtype).to(device)
        #     # new_net_input = torch.unsqueeze(new_net_input, 0)
        #     out_mod_np = net(new_net_input).detach().cpu().numpy()[0]
        #
        #     X_out_mod_a, X_out_mod_b = out_mod_np[0], out_mod_np[1]
        #     #X_out_mod_a, X_out_mod_b = out_mod_np[0,0], out_mod_np[0,1]
        #
        #     reconstructed_spectrogram_mod = X_out_mod_a + 1j*X_out_mod_b
        #     reconstructed_audio_mod = librosa.istft(reconstructed_spectrogram_mod)
        #
        #     plt.figure(figsize=(14,10))
        #     librosa.display.specshow(librosa.amplitude_to_db(reconstructed_spectrogram_mod), sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        #     plt.colorbar()
        #     plt.savefig(inner_path + '/magnitude_mod.png')
        #
        #     sf.write(inner_path + '/audio_mod.wav', reconstructed_audio_mod, sr, 'PCM_24')

        ######################

        plt.close('all')

        print('\n[DONE]\n')

    print('\n[COMPLETED]\n')



if __name__ == '__main__':

    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help='Source directory or file', required=True)
    parser.add_argument('--results_dir', type=str, default='./results', help='Results Directory')
    parser.add_argument('--n_audio', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--act_fun', type=str, default='LeakyReLU', help='Activation function')
    parser.add_argument('--net', type=str, default='./nets/default.json', help='Path of json file with net specifications', required=True)
    parser.add_argument('--input_depth', type=int, default=2, help='CNN input depth')

    parser.add_argument('--gap_length', type=int, default=3, help='Dimension of gaps to inpaint in time bin')
    parser.add_argument('--ngaps', type=int, default=1, help='Number of gaps to inpaint')
    parser.add_argument('--audio_len', type=int, default=5, help='Length of audio sample')
    parser.add_argument('--offset', type=int, default=5, help='Audio offset length')
    parser.add_argument('--tot_gap', type=int, default=500, help='Total gap duration')
    parser.add_argument('--min_gap', type=int, default=40, help='Minimun dimension of gap')
    parser.add_argument('--max_gap', type=int, default=80, help='Maximum dimension of gap')
    parser.add_argument('--context', type=int, default=150, help='Dimension of context between two contiguous gaps')

    parser.add_argument('--input_noise', type=str, default='uniform', help='Type of noise given to the net as input: uniform|normal|pink|brown')
    parser.add_argument('--gpu', type=str, default='0', help='Number of GPU device')
    parser.add_argument('--layers', action='store_true', help='Save mid layers representations')
    parser.add_argument('--mod', action='store_true', help='Save modified output')
    parser.add_argument('--output_every', type=int, default=0, help='Save output of net every n epochs')
    parser.add_argument('--noise', type=int, default=0, help='1/f noise exponent')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1.0)

    args = parser.parse_args()

    # wandb_args = {
    #     'alpha': args.alpha,
    #     'beta': args.beta,
    #     'gamma': args.gamma,
    #     'delta': args.delta
    # }
    #
    # wandb.init(config=args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0')

    main(args, device)
