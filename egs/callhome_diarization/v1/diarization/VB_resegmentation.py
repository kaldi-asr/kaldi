#!/usr/bin/env python3

# Copyright 2019  Zili Huang

# This script is evoked by diarization/VB_resegmentation.sh. It prepares the necessary
# inputs for the VB system and creates the output RTTM file. The inputs include data directory
# (data_dir), the rttm file to initialize the VB system(init_rttm_filename), the directory to
# output the rttm prediction(output_dir), path to diagonal UBM model(dubm_model) and path to 
# i-vector extractor model(ie_model).

import numpy as np
import VB_diarization
import kaldi_io
import argparse
from convert_VB_model import load_dubm, load_ivector_extractor 

def get_utt_list(utt2spk_filename):
    with open(utt2spk_filename, 'r') as fh:
        content = fh.readlines()
    utt_list = [line.split()[0] for line in content]
    print("{} utterances in total".format(len(utt_list)))
    return utt_list

# prepare utt2num_frames dictionary
def get_utt2num_frames(utt2num_frames_filename):
    utt2num_frames = {}
    with open(utt2num_frames_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames

# prepare utt2feats dictionary
def get_utt2feats(utt2feats_filename):
    utt2feats = {}
    with open(utt2feats_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        utt2feats[line_split[0]] = line_split[1]
    return utt2feats

def create_ref(uttname, utt2num_frames, full_rttm_filename):
    num_frames = utt2num_frames[uttname]

    # We use 0 to denote silence frames and 1 to denote overlapping frames.
    ref = np.zeros(num_frames)
    speaker_dict = {}
    num_spk = 0

    with open(full_rttm_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname_line = line_split[1]
        if uttname != uttname_line:
            continue
        start_time, duration = int(float(line_split[3]) * 100), int(float(line_split[4]) * 100)
        end_time = start_time + duration
        spkname = line_split[7]
        if spkname not in speaker_dict.keys():
            spk_idx = num_spk + 2
            speaker_dict[spkname] = spk_idx
            num_spk += 1
        
        for i in range(start_time, end_time):
            if i < 0:
                raise ValueError("Time index less than 0")
            elif i >= num_frames:
                print("Time index exceeds number of frames")
                break
            else:
                if ref[i] == 0:
                    ref[i] = speaker_dict[spkname] 
                else:
                    ref[i] = 1 # The overlapping speech is marked as 1.
    return ref.astype(int)

# create output rttm file
def create_rttm_output(uttname, predicted_label, output_dir, channel):
    num_frames = len(predicted_label)

    start_idx = 0
    seg_list = []

    last_label = predicted_label[0]
    for i in range(num_frames):
        if predicted_label[i] == last_label: # The speaker label remains the same.
            continue
        else: # The speaker label is different.
            if last_label != 0: # Ignore the silence.
                seg_list.append([start_idx, i, last_label])
            start_idx = i
            last_label = predicted_label[i]
    if last_label != 0:
        seg_list.append([start_idx, num_frames, last_label])

    with open("{}/{}_predict.rttm".format(output_dir, uttname), 'w') as fh:
        for i in range(len(seg_list)):
            start_frame = (seg_list[i])[0]
            end_frame = (seg_list[i])[1]
            label = (seg_list[i])[2]
            duration = end_frame - start_frame
            fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(uttname, channel, start_frame / 100.0, duration / 100.0, label))
    return 0

def main():
    parser = argparse.ArgumentParser(description='VB Resegmentation Wrapper')
    parser.add_argument('data_dir', type=str, help='Subset data directory')
    parser.add_argument('init_rttm_filename', type=str, 
                        help='The rttm file to initialize the VB system, usually the AHC cluster result')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('dubm_model', type=str, help='Path of the diagonal UBM model')
    parser.add_argument('ie_model', type=str, help='Path of the i-vector extractor model')

    parser.add_argument('--max-speakers', type=int, default=10,
                        help='Maximum number of speakers expected in the utterance (default: 10)')
    parser.add_argument('--max-iters', type=int, default=10,
                        help='Maximum number of algorithm iterations (default: 10)')
    parser.add_argument('--downsample', type=int, default=25,
                        help='Perform diarization on input downsampled by this factor (default: 25)')
    parser.add_argument('--alphaQInit', type=float, default=100.0,
                        help='Dirichlet concentraion parameter for initializing q')
    parser.add_argument('--sparsityThr', type=float, default=0.001,
                        help='Set occupations smaller that this threshold to 0.0 (saves memory as \
                        the posteriors are represented by sparse matrix)')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='Stop iterating, if obj. fun. improvement is less than epsilon')
    parser.add_argument('--minDur', type=int, default=1,
                        help='Minimum number of frames between speaker turns imposed by linear \
                        chains of HMM states corresponding to each speaker. All the states \
                        in a chain share the same output distribution')
    parser.add_argument('--loopProb', type=float, default=0.9,
                        help='Probability of not switching speakers between frames')
    parser.add_argument('--statScale', type=float, default=0.2,
                        help='Scale sufficient statiscits collected using UBM')
    parser.add_argument('--llScale', type=float, default=1.0,
                        help='Scale UBM likelihood (i.e. llScale < 1.0 make atribution of \
                        frames to UBM componets more uncertain)')
    parser.add_argument('--channel', type=int, default=0,
                        help='Channel information in the rttm file')
    parser.add_argument('--initialize', type=int, default=1,
                        help='Whether to initalize the speaker posterior')

    args = parser.parse_args()
    print(args)

    utt_list = get_utt_list("{}/utt2spk".format(args.data_dir))
    utt2num_frames = get_utt2num_frames("{}/utt2num_frames".format(args.data_dir))
    
    # Load the diagonal UBM and i-vector extractor
    dubm_para = load_dubm(args.dubm_model)
    ie_para = load_ivector_extractor(args.ie_model)

    # Check the diagonal UBM and i-vector extractor model
    assert '<WEIGHTS>' in dubm_para and '<MEANS_INVVARS>' in dubm_para and '<INV_VARS>' in dubm_para
    DUBM_WEIGHTS, DUBM_MEANS_INVVARS, DUBM_INV_VARS = dubm_para['<WEIGHTS>'], dubm_para['<MEANS_INVVARS>'], dubm_para['<INV_VARS>']
    assert 'M' in ie_para
    IE_M = np.transpose(ie_para['M'], (2, 0, 1))
    
    m = DUBM_MEANS_INVVARS / DUBM_INV_VARS
    iE = DUBM_INV_VARS
    w = DUBM_WEIGHTS
    V = IE_M

    # Load the MFCC features
    feats_dict = get_utt2feats("{}/feats.scp".format(args.data_dir))

    for utt in utt_list:
        # Get the alignments from the clustering result.
        # In init_ref, 0 denotes the silence silence frames
        # 1 denotes the overlapping speech frames, the speaker
        # label starts from 2.
        init_ref = create_ref(utt, utt2num_frames, args.init_rttm_filename)

        # load MFCC features
        X = kaldi_io.read_mat(feats_dict[utt]).astype(np.float64)
        assert len(init_ref) == len(X)

        # Keep only the voiced frames (0 denotes the silence 
        # frames, 1 denotes the overlapping speech frames).
        mask = (init_ref >= 2)
        X_voiced = X[mask]
        init_ref_voiced = init_ref[mask] - 2

        if X_voiced.shape[0] == 0:
            print("Warning: {} has no voiced frames in the initialization file".format(utt))
            continue

        # Initialize the posterior of each speaker based on the clustering result.
        if args.initialize:
            q = VB_diarization.frame_labels2posterior_mx(init_ref_voiced, args.max_speakers)
        else:
            q = None
        
        # VB resegmentation

        # q  - S x T matrix of posteriors attribution each frame to one of S possible
        #      speakers, where S is given by opts.maxSpeakers
        # sp - S dimensional column vector of ML learned speaker priors. Ideally, these
        #      should allow to estimate # of speaker in the utterance as the
        #      probabilities of the redundant speaker should converge to zero.
        # Li - values of auxiliary function (and DER and frame cross-entropy between q
        #      and reference if 'ref' is provided) over iterations.
        q_out, sp_out, L_out = VB_diarization.VB_diarization(X_voiced, m, iE, w, V, sp=None, q=q, maxSpeakers=args.max_speakers, maxIters=args.max_iters, VtiEV=None,
                                  downsample=args.downsample, alphaQInit=args.alphaQInit, sparsityThr=args.sparsityThr, epsilon=args.epsilon, minDur=args.minDur,
                                  loopProb=args.loopProb, statScale=args.statScale, llScale=args.llScale, ref=None, plot=False)
        predicted_label_voiced = np.argmax(q_out, 1) + 2
        predicted_label = (np.zeros(len(mask))).astype(int)
        predicted_label[mask] = predicted_label_voiced

        # Create the output rttm file
        create_rttm_output(utt, predicted_label, args.output_dir, args.channel)
    return 0

if __name__ == "__main__":
    main()
