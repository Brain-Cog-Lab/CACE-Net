# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from tqdm import tqdm
from pytorch.models import *
import librosa
from utils.config import classes_num as num
from pytorch.pytorch_utils import move_data_to_device

class CLIP(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, I_dim, A_dim, hidden_dim, T, batchsize, args):
        """
        T: softmax temperature (default: 0.07)
        """
        super(CLIP, self).__init__()

        self.T = T
        self.batchsize = batchsize

        # weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        # I_model = efficientnet_v2_m(weights=weights, progress = True)
        weights = VGG19_BN_Weights.IMAGENET1K_V1
        I_model = vgg19_bn(weights=weights)
        I_model.train()
        I_model = torch.nn.Sequential(*(list(I_model.children())[:-1]))

        # sample_rate = args.sample_rate
        # window_size = args.window_size
        # hop_size = args.hop_size
        # mel_bins = args.mel_bins
        # fmin = args.fmin
        # fmax = args.fmax
        # model_type = args.model_type
        # classes_num = num
        # Model = eval(model_type)
        # A_model = Model(sample_rate=sample_rate, window_size=window_size, 
        # hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        # classes_num=classes_num)
        # A_model.train()
        device = torch.device('cuda')
        # checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # A_model.load_state_dict(checkpoint['model'])

        # create the encoders
        self.encoder_I = I_model
        # self.encoder_A = A_model
        self.encoder_I.to(device)
        # self.encoder_A.to(device)
        # for param_A in self.encoder_A.parameters():
        #     param_A.requires_grad = False  # not update by gradient
        self.alignment_I = nn.Linear(I_dim, hidden_dim, bias=False)
        self.alignment_A = nn.Linear(A_dim, hidden_dim, bias=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, images, audio, args):
        """
        Input:
            image: a batch of images [b ,t, 3, 224, 224]
            audio: a batch of audios [b, 10, 32000]
        Output:
            logits, targets
        """

        B, T, C, H, W = images.shape
        I_feature = self.encoder_I(images.reshape(B*T, C, H, W))  # [b, t, 512, 7, 7]

        _, C, H, W = I_feature.shape
        I_feature = torch.mean(I_feature.reshape(B, T, C, H, W), dim=1)
        I_feature = self.pool(I_feature).squeeze()  # [B, 512]

        # compute audio features
        # with torch.no_grad():
        #     batch_output_dict = self.encoder_A(audio, None)
        # if 'embedding' in batch_output_dict.keys():
        #     A_feature = batch_output_dict['embedding']  # [B, 128]

        A_feature = torch.mean(audio.reshape(args.batch_size, 10, 128), dim=1)
        # dimension alignment
        I_f = nn.functional.normalize(self.alignment_I(I_feature), dim=1)
        A_f = nn.functional.normalize(self.alignment_A(A_feature), dim=1)

        # compute logits: NxN
        logits = torch.matmul(I_f,A_f.T)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.arange(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels