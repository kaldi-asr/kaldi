#!/usr/bin/env python
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0
# Works with both python2 and python3

import numpy as np
import soundfile as sf
import time
import os, errno
from tqdm import tqdm
import argparse

from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft 
from nara_wpe import project_root

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', nargs='+')
args = parser.parse_args()

input_files = args.files[:len(args.files)//2]
output_files = args.files[len(args.files)//2:]
out_dir = os.path.dirname(output_files[0])
try: 
    os.makedirs(out_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

stft_options = dict(
    size=512,
    shift=128,
    window_length=None,
    fading=True,
    pad=True,
    symmetric_window=False
)

sampling_rate = 16000
delay = 3
iterations = 5
taps = 10

signal_list = [
    sf.read(f)[0]
    for f in input_files
]
y = np.stack(signal_list, axis=0)
Y = stft(y, **stft_options).transpose(2, 0, 1)
Z = wpe(Y, iterations=iterations, statistics_mode='full').transpose(1, 2, 0)
z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])

for d in range(len(signal_list)):
    sf.write(output_files[d], z[d,:], sampling_rate)
