import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from models import *
from pytorch_utils import move_data_to_device
import config

parser_at = argparse.ArgumentParser(description='Example of parser. ')
parser_at.add_argument('--sample_rate', type=int, default=32000)
parser_at.add_argument('--window_size', type=int, default=1024)
parser_at.add_argument('--hop_size', type=int, default=320)
parser_at.add_argument('--mel_bins', type=int, default=64)
parser_at.add_argument('--fmin', type=int, default=50)
parser_at.add_argument('--fmax', type=int, default=14000) 
parser_at.add_argument('--model_type', type=str, required=True)
parser_at.add_argument('--checkpoint_path', type=str, required=True)
parser_at.add_argument('--audio_path', type=str)
parser_at.add_argument('--cuda', action='store_true', default=False)
args = parser_at.parse_args()

def audio_tagging(filename, args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    classes_num = config.classes_num

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')


    audio_path = "/home/hexiang/Encoders/data/AVE_audio/AVE_wav/" + filename
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.train()
        batch_output_dict = model(waveform, None)

    # clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
    # """(classes_num,)"""

    # for i in range(len(clipwise_output)):
    #     print("---------------{}-th-------------".format(i))
    #     clipwise_output_i = clipwise_output[i]
    #     sorted_indexes = np.argsort(clipwise_output_i)[::-1]
    #
    #     # Print audio tagging top probabilities
    #     for k in range(3):
    #         print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
    #             clipwise_output_i[sorted_indexes[k]]))


    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu()

    return embedding.astype(torch.double())

if __name__ == '__main__':
    audio_tagging(filename="---1_cCGK4M.wav", args=args)