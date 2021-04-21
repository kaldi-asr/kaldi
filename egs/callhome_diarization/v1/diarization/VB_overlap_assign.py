#!/usr/bin/env python3
# Copyright 2019 JSALT (Author: Latané Bullock)
# Apache 2.0

# This script is modified from VB_resegmentation.py. It uses the speaker attribution
# (q) matrix from VB resegmentation and an overlap hypothesis RTTM to assign
# speakers to overlapped frames.

import numpy as np
import argparse


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

def rttm2one_hot(uttname, utt2num_frames, full_rttm_filename):
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
                print("Time index exceeds number of frames. Cropping RTTM from {} to {} \
                      frames".format(end_time, num_frames))
                break
            else:
                if ref[i] == 0:
                    ref[i] = speaker_dict[spkname]
                else:
                    ref[i] = 1 # The overlapping speech is marked as 1.
    return ref.astype(int)

# create output rttm file
def create_rttm_output(uttname, pri_sec, predicted_label, output_dir, channel):
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

    with open("{}/{}_ovl_{}.rttm".format(output_dir, uttname, pri_sec), 'w') as fh:
        for i in range(len(seg_list)):
            start_frame = (seg_list[i])[0]
            end_frame = (seg_list[i])[1]
            label = (seg_list[i])[2]
            duration = end_frame - start_frame
            fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(
                uttname, channel, start_frame / 100.0, duration / 100.0, label))
    return 0



def main():
    parser = argparse.ArgumentParser(description='Frame-level overlap reassignment with speaker posterior attributions')
    parser.add_argument('data_dir', type=str, help='Kaldi-style data directory')
    parser.add_argument('ovl_rttm', type=str, help='RTTM file with overlap segments')
    parser.add_argument('init_rttm', type=str, help='Initial RTTM file (to get voiced segments)')
    parser.add_argument('ovl_dir', type=str, help='Working directory for VB resegmentation')

    args = parser.parse_args()
    print(args)

    utt_list = get_utt_list("{}/utt2spk".format(args.data_dir))
    utt2num_frames = get_utt2num_frames("{}/utt2num_frames".format(args.data_dir))

    for utt in utt_list:
        n_frames = utt2num_frames[utt]
        
        vad = rttm2one_hot(utt, utt2num_frames, args.init_rttm)
        overlap = rttm2one_hot(utt, utt2num_frames, args.ovl_rttm)

        # Keep only the voiced frames (0 denotes the silence
        # frames, 1 denotes the overlapping speech frames).
        mask = (vad >= 2)
        print ("Recording: {}, Num frames: {}, voiced: {}".format(utt, n_frames, mask.sum()))
        
        # Remember: q is only for voiced frames
        q_out = np.load('{}/{}_q_out.npy'.format(args.ovl_dir, utt))
        
        # Standard procedure from VB reseg - take the most likely speaker
        predicted_label_voiced = np.argsort(-q_out, 1)[:,0] + 2
        predicted_label = (np.zeros(n_frames)).astype(int)
        predicted_label[mask] = predicted_label_voiced
        # This is the 'primary' speaker for each frame
        create_rttm_output(utt, 'pri', predicted_label, args.ovl_dir, channel=1)

        # Write "secondary" speakers for overlap regions
        # -- take second most likely speaker for each overlap frame
        predicted_label_voiced = np.argsort(-q_out, 1)[:,1] + 2
        predicted_label = (np.zeros(n_frames)).astype(int)
        predicted_label[mask] = predicted_label_voiced
        predicted_label[np.where(overlap==0)[0]] = 0

        create_rttm_output(utt, 'sec', predicted_label, args.ovl_dir, channel=1)

    return 0


if __name__ == "__main__":
    main()