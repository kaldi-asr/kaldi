#!/usr/bin/env python
# Copyright 2020 Johns Hopkins University (Author: Bar Ben-Yair)
# Apache 2.0
# Works with both python2 and python3
# This script assumes that WPE (nara_wpe) is installed locally using miniconda.
# ../../../tools/extras/install_miniconda.sh and ../../../tools/extras/install_wpe.sh
# needs to be run and this script needs to be launched run with that version of
# python.
# See local/run_wpe.sh for example.
# (Updated to work with LibriCSS data)

import numpy as np
import soundfile as sf
import time
import os, errno
from tqdm import tqdm
import argparse

# to avoid huge memory consumption we decided to use `online_wpe` instead of the offline one
# following the advice from Christoph Boeddeker at Paderborn University
from nara_wpe.wpe import wpe_v8 as wpe,OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe import project_root

# Input is a list of wav files corresponding to all the channels of a single recording, and
# and output directory path where the dereverberated wavs will be stored.
parser = argparse.ArgumentParser()
parser.add_argument('in_wavs')
parser.add_argument('out_dir')
args = parser.parse_args()

# Create output directory
try:
    os.makedirs(args.out_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Read all input channels
in_wavs = []
with open(args.in_wavs,'r') as f:
    for line in f.readlines():
        in_wavs.append(line.strip())

stft_options = dict(
    size=512,
    shift=128,
    window_length=None,
    fading=True,
    pad=True,
    symmetric_window=False
)

channels=len(in_wavs)
sampling_rate = 16000
delay = 3
iterations = 5
taps = 10
alpha= 0.9999

def aquire_framebuffer():
    buffer = list(Y[:taps+delay+1, :, :])
    for t in range(taps+delay+1, T):
        arr = np.array(buffer)
        yield arr
        buffer.append(Y[t, :, :])
        buffer.pop(0)

# Read all input channels
signal_list = [sf.read(in_wav,dtype='int16')[0] for in_wav in in_wavs]

# Perform processing on stacked channel input
signal_list_len=len(signal_list)
y = np.stack(signal_list, axis=0)
del signal_list
Y = stft(y, **stft_options).transpose(1, 2, 0)
del y
    
T, _, _ = Y.shape
Z_list = []
    
online_wpe = OnlineWPE(
	taps=taps,
	delay=delay,
	alpha=alpha,
	frequency_bins=Y.shape[1],
	channel=channels
)
    
for Y_step in tqdm(aquire_framebuffer()):
    if np.sum(Y_step.flatten())!=0:
        Z_list.append(online_wpe.step_frame(Y_step))
    else:
        Z_list.append(Y_step[0,:,:].reshape((Y_step.shape[1],Y_step.shape[2])))
del Y 
Z = np.asarray(np.stack(Z_list)).transpose(2, 0, 1)
z = istft(Z, size=stft_options['size'], shift=stft_options['shift']).astype('int16')
del Z

# Write dereverberated output
for d in range(signal_list_len):
    filename = os.path.basename(in_wavs[d])
    out_path = os.path.join(args.out_dir,filename)
    sf.write(out_path,z[d,:],sampling_rate)