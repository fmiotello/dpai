import torch
from torch import nn
from .base import get_activation, conv, conv3d, conv2dbn, conv3dbn, Concat, Concat3D

__all__ = [
    "MulResUnet",
    "MulResUnet3D",
]

class Block3d(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True, drop=0.):
        super(Block3d, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = conv3dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                 bias=bias, act_fun=act_fun)
        self.conv3x3 = conv3dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv5x5 = conv3dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv7x7 = conv3dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.bn1 = nn.BatchNorm3d(self.out_dim)
        self.bn2 = nn.BatchNorm3d(self.out_dim)
        self.act = get_activation(act_fun)
        self.dr = nn.Dropout3d(drop)

    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = torch.cat([out1, out2, out3], axis=1)
        out = self.bn1(out)
        out = self.dr(out)
        out = torch.add(self.shortcut(input), out)
        out = self.act(out)
        out = self.bn2(out)
        out = self.dr(out)
        return out


class ResPath3d(nn.Module):
    def __init__(self, f_in, f_out, act_fun='LeakyReLU', bias=True, drop=0.):
        super(ResPath3d, self).__init__()
        self.conv3x3 = conv3dbn(f_in, f_out, 3, 1, bias=bias, act_fun=act_fun)
        self.conv1x1 = conv3dbn(f_in, f_out, 1, 1, bias=bias, act_fun=act_fun)
        self.bn = nn.BatchNorm3d(f_out)
        self.act = get_activation(act_fun)
        self.dr = nn.Dropout3d(drop)

    def forward(self, input):
        out = torch.add(self.conv1x1(input), self.conv3x3(input))
        out = self.act(out)
        out = self.bn(out)
        out = self.dr(out)
        return out


def MulResUnet3D(num_input_channels=1,
                 num_output_channels=1,
                 num_channels_down=[16, 32, 64, 128, 256],
                 num_channels_up=[16, 32, 64, 128, 256],
                 num_channels_skip=[16, 32, 64, 128],
                 alpha=1.67,
                 last_act_fun=None,
                 need_bias=True,
                 upsample_mode='nearest',
                 act_fun='LeakyReLU',
                 dropout=0.):
    assert len(num_channels_down) == len(num_channels_up) == (len(num_channels_skip) + 1)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model
    multires = Block3d(num_channels_down[0], num_input_channels,
                       alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)

    model_tmp.add(multires)
    input_depth = multires.out_dim

    for i in range(1, n_scales):

        deeper = nn.Sequential()
        skip = nn.Sequential()
        # add the multi-res block for encoder
        multires = Block3d(num_channels_down[i], input_depth,
                           alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
        # add the stride downsampling for encoder
        deeper.add(conv3d(input_depth, input_depth, 3, stride=2, bias=need_bias))
        deeper.add(nn.BatchNorm3d(input_depth))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout3d(dropout))
        deeper.add(multires)

        if num_channels_skip[i - 1] != 0:
            # add the Path residual block with skip-connection
            skip.add(ResPath3d(input_depth, num_channels_skip[i - 1], act_fun=act_fun, bias=need_bias, drop=dropout))
            model_tmp.add(Concat3D(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        deeper_main = nn.Sequential()

        if i != len(num_channels_down) - 1:
            deeper.add(deeper_main)
        # add the upsampling to decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add the multi-res block to decoder
        model_tmp.add(Block3d(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1],
                              alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout))

        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = num_channels_up[0] * alpha
    last_kernel = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    # convolutional filter for output
    model.add(conv3d(last_kernel, num_output_channels, 3, bias=need_bias))

    if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
        last_act_fun = None
    if last_act_fun is not None:
        model.add(get_activation(last_act_fun))

    return model

# class Block2d(nn.Module):
#     def __init__(self, U, f_in, conv_type='conv', anchor=[1, 2, 3], alpha=1.67, act_fun='LeakyReLU', bias=True, drop=0.):
#         super(Block2d, self).__init__()
#         W = alpha * U
#         self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
#         self.shortcut = conv2dbn(conv_type, f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, bias=bias, act_fun=act_fun, anchor=1)
#         self.conv3x3 = conv2dbn(conv_type, f_in, int(W * 0.167), 3, 1, bias=bias, act_fun=act_fun, anchor=anchor[0])
#         self.conv5x5 = conv2dbn(conv_type, int(W * 0.167), int(W * 0.333), 5, 1, bias=bias, act_fun=act_fun, anchor=anchor[1])
#         self.conv7x7 = conv2dbn(conv_type, int(W * 0.333), int(W * 0.5), 7, 1, bias=bias, act_fun=act_fun, anchor=anchor[2])
#         self.dr = nn.Dropout2d(drop)
#         self.act = get_activation(act_fun)
#
#     def forward(self, input):
#         out1 = self.conv3x3(input)
#         out2 = self.conv5x5(out1)
#         out3 = self.conv7x7(out2)
#         out = torch.cat([out1, out2, out3], axis=1)
#         #out = out1 + out2 + out3
#         out = self.dr(out)
#         out = torch.add(self.shortcut(input), out)
#         out = self.act(out)
#         out = self.dr(out)
#         return out
#
#
# class ResPath2d(nn.Module):
#     def __init__(self, f_in, f_out, length, conv_type='conv', anchor=[1, 2], act_fun='LeakyReLU', bias=True, drop=0.):
#         super(ResPath2d, self).__init__()
#         self.dr = nn.Dropout2d(drop)
#         self.net = []
#         self.net.append(conv2dbn(conv_type, f_in, f_out, 3, 1, bias=bias, act_fun=act_fun, anchor=anchor[1]))
#         self.net.append(conv2dbn(conv_type, f_in, f_out, 1, 1, bias=bias, act_fun=act_fun, anchor=anchor[0]))
#         self.net.append(nn.BatchNorm2d(f_out))
#         self.net.append(self.dr)
#
#         for i in range(length - 1):
#             self.net.append(conv2dbn(conv_type, f_out, f_out, 3, 1, bias=bias, act_fun=act_fun, anchor=anchor[1]))
#             self.net.append(conv2dbn(conv_type, f_out, f_out, 1, 1, bias=bias, act_fun=act_fun, anchor=anchor[0]))
#             self.net.append(nn.BatchNorm2d(f_out))
#             self.net.append(self.dr)
#
#         self.act = get_activation(act_fun)
#         self.length = length
#         self.net = nn.Sequential(*self.net)
#
#     def forward(self, input):
#         out = self.net[2](self.dr(self.act(torch.add(self.net[0](input), self.net[1](input)))))
#         for i in range(1, self.length):
#             out = self.net[i * 3 + 2](self.dr(self.act(torch.add(self.net[i * 3](out), self.net[i * 3 + 1](out)))))
#
#         return out
#
#
# def MulResUnet(num_input_channels=1,
#                num_output_channels=1,
#                num_channels_down=[16, 32, 64, 128, 256],
#                num_channels_up=[16, 32, 64, 128, 256],
#                num_channels_skip=[16, 32, 64, 128],
#                alpha=1.67,
#                last_act_fun=None,
#                need_bias=True,
#                upsample_mode='nearest',
#                act_fun='LeakyReLU',
#                dropout=0.):
#     assert len(num_channels_down) == len(num_channels_up) == (len(num_channels_skip) + 1)
#
#     n_scales = len(num_channels_down)
#
#     if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
#         upsample_mode = [upsample_mode] * n_scales
#
#     model = nn.Sequential()
#     model_tmp = model
#     multires = Block2d(num_channels_down[0], num_input_channels, conv_type='conv', anchor=[1, 2, 3],
#                        alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
#
#     model_tmp.add(multires)
#     input_depth = multires.out_dim
#
#     for i in range(1, n_scales):
#
#         deeper = nn.Sequential()
#         skip = nn.Sequential()
#         # multi-res Block in the encoders
#         multires = Block2d(num_channels_down[i], input_depth, conv_type='conv', anchor=[i, 1+i, 2+i],
#                            alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
#         # stride downsampling
#         deeper.add(conv('conv', input_depth, input_depth, 3, stride=2, bias=need_bias))
#         deeper.add(get_activation(act_fun))
#         deeper.add(nn.Dropout2d(dropout))
#         deeper.add(multires)
#
#         if num_channels_skip[i - 1] != 0:
#             # add the path residual block, note that the number of filters is set to 1
#             skip.add(ResPath2d(input_depth, num_channels_skip[i - 1], 1, conv_type='conv', anchor=[i, 1+i],
#                                 act_fun=act_fun, bias=need_bias, drop=dropout))
#             model_tmp.add(Concat(1, skip, deeper))
#         else:
#             model_tmp.add(deeper)
#
#         deeper_main = nn.Sequential()
#
#         if i != len(num_channels_down) - 1:
#             # not the deepest
#             deeper.add(deeper_main)
#         # add upsampling to the decoder
#         deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
#         # add multi-res block to the decoder
#         model_tmp.add(Block2d(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1], conv_type='conv', anchor=[1, 2, 3],
#                               alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout))
#
#         input_depth = multires.out_dim
#         model_tmp = deeper_main
#
#     # add the convolutional filter for output
#     W = num_channels_up[0] * alpha
#     model.add(conv('conv', int(W * 0.167) + int(W * 0.333) + int(W * 0.5), num_output_channels, 1, bias=need_bias))
#
#     if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
#         last_act_fun = None
#     if last_act_fun is not None:
#         model.add(get_activation(last_act_fun))
#
#     return model

class Block2d(nn.Module):
    def __init__(self, U, f_in, kernel_size=[3,3,3], conv_type=['conv', 'conv', 'conv'], anchor=[1, 2, 3], alpha=1.67, act_fun='LeakyReLU', bias=True, drop=0.):
        super(Block2d, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = conv2dbn(conv_type[0], f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, bias=bias, act_fun=act_fun, anchor=1)
        self.conv3x3 = conv2dbn(conv_type[0], f_in, int(W * 0.167), kernel_size[0], 1, dilation=1, bias=bias, act_fun=act_fun, anchor=anchor[0])
        self.conv5x5 = conv2dbn(conv_type[1], int(W * 0.167), int(W * 0.333), kernel_size[1], 1, dilation=1, bias=bias, act_fun=act_fun, anchor=anchor[1])
        self.conv7x7 = conv2dbn(conv_type[2], int(W * 0.333), int(W * 0.5), kernel_size[2], 1, dilation=1, bias=bias, act_fun=act_fun, anchor=anchor[2])
        self.dr = nn.Dropout2d(drop)
        self.act = get_activation(act_fun)

    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)

        out = torch.cat([out1, out2, out3], axis=1)

        out = self.dr(out)
        out = torch.add(self.shortcut(input), out)
        out = self.act(out)
        out = self.dr(out)
        return out


class ResPath2d(nn.Module):
    def __init__(self, f_in, f_out, length, kernel_size=3, conv_type=['conv', 'conv'], anchor=[1, 1], act_fun='LeakyReLU', bias=True, drop=0.):
        super(ResPath2d, self).__init__()
        self.dr = nn.Dropout2d(drop)
        self.net = []
        self.net.append(conv2dbn(conv_type[0], f_in, f_out, kernel_size, 1, bias=bias, act_fun=act_fun, anchor=anchor[0]))
        self.net.append(conv2dbn(conv_type[1], f_in, f_out, 1, 1, bias=bias, act_fun=act_fun, anchor=anchor[1]))
        self.net.append(nn.BatchNorm2d(f_out))
        self.net.append(self.dr)

        for i in range(length - 1):
            self.net.append(conv2dbn(conv_type[0], f_out, f_out, kernel_size, 1, bias=bias, act_fun=act_fun, anchor=anchor[0]))
            self.net.append(conv2dbn(conv_type[1], f_out, f_out, 1, 1, bias=bias, act_fun=act_fun, anchor=anchor[1]))
            self.net.append(nn.BatchNorm2d(f_out))
            self.net.append(self.dr)

        self.act = get_activation(act_fun)
        self.length = length
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.net[2](self.dr(self.act(torch.add(self.net[0](input), self.net[1](input)))))
        for i in range(1, self.length):
            out = self.net[i * 3 + 2](self.dr(self.act(torch.add(self.net[i * 3](out), self.net[i * 3 + 1](out)))))

        return out


def MulResUnet(num_input_channels=1,
               num_output_channels=1,
               num_channels_down=[16, 32, 64, 128, 256],
               num_channels_up=[16, 32, 64, 128, 256],
               num_channels_skip=[16, 32, 64, 128],
               kernel_size_down=[3, 3, 3, 3, 3],
               kernel_size_up=[3, 3, 3, 3, 3],
               kernel_size_skip=[3, 3, 3, 3],
               kernel_size_downsampling=[3, 3, 3, 3],
               conv_type_down='conv',
               conv_type_up='conv',
               conv_type_res='conv',
               conv_type_downsampling='conv',
               conv_type_out='conv',
               anchor_down=1,
               anchor_up=1,
               anchor_res=1,
               anchor_downsampling=1,
               anchor_out=1,
               alpha=1.67,
               last_act_fun=None,
               need_bias=True,
               upsample_mode='nearest',
               act_fun='ELU',
               dropout=0.):

    assert len(num_channels_down) == len(num_channels_up) == (len(num_channels_skip) + 1)
    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    model = nn.Sequential()
    model_tmp = model
    multires = Block2d(num_channels_down[0], num_input_channels, kernel_size_down[0], conv_type=conv_type_down[0], anchor=anchor_down[0],
                       alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)

    model_tmp.add(multires)
    input_depth = multires.out_dim

    for i in range(1, n_scales):

        deeper = nn.Sequential()
        skip = nn.Sequential()
        # multi-res Block in the encoders
        multires = Block2d(num_channels_down[i], input_depth, kernel_size_down[i], conv_type=conv_type_down[i], anchor=anchor_down[i],
                           alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
        # stride downsampling
        deeper.add(conv(conv_type_downsampling[i-1], input_depth, input_depth, kernel_size_downsampling[i-1], stride=2, bias=need_bias, anchor=anchor_downsampling[i-1]))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout2d(dropout))
        deeper.add(multires)

        if num_channels_skip[i - 1] != 0:
            # add the path residual block, note that the number of filters is set to 1
            skip.add(ResPath2d(input_depth, num_channels_skip[i - 1], 1, kernel_size_skip[i-1], conv_type=conv_type_res[i-1], anchor=anchor_res[i-1],
                                act_fun=act_fun, bias=need_bias, drop=dropout))
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        deeper_main = nn.Sequential()

        if i != len(num_channels_down) - 1:
            # not the deepest
            deeper.add(deeper_main)
        # add upsampling to the decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add multi-res block to the decoder
        model_tmp.add(Block2d(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1], kernel_size_up[i-1], conv_type=conv_type_up[i-1], anchor=anchor_up[i-1],
                              alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout))

        input_depth = multires.out_dim
        model_tmp = deeper_main

    # add the convolutional filter for output
    W = num_channels_up[0] * alpha
    model.add(conv(conv_type_out, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), num_output_channels, 1, bias=need_bias, anchor=anchor_out))

    if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
        last_act_fun = None
    if last_act_fun is not None:
        model.add(get_activation(last_act_fun))

    return model
