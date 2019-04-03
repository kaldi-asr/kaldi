#!/usr/bin/env python

import numpy as np
import VB_diarization
import pickle
import kaldi_io
import sys
import argparse
import commands

def get_utt_list(utt2spk_filename):
    utt_list = []
    with open(utt2spk_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt_list.append(line_split[0])
    print("{} UTTERANCES IN TOTAL".format(len(utt_list)))
    return utt_list

def utt_num_frames_mapping(utt2num_frames_filename):
    utt2num_frames = {}
    with open(utt2num_frames_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames

def create_ref_file(uttname, utt2num_frames, full_rttm_filename, temp_dir, rttm_filename):
    utt_rttm_file = open("{}/{}".format(temp_dir, rttm_filename), 'w')

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
        else:
            utt_rttm_file.write(line + "\n")
        start_time = int(float(line_split[3]) * 100)
        duration_time = int(float(line_split[4]) * 100)
        end_time = start_time + duration_time
        spkname = line_split[7]
        if spkname not in speaker_dict.keys():
            spk_idx = num_spk + 2
            speaker_dict[spkname] = spk_idx
            num_spk += 1
        
        for i in range(start_time, end_time):
            if i < 0:
                raise ValueError(line)
            elif i >= num_frames:
                print("{} EXCEED NUM_FRAMES".format(line))
                break
            else:
                if ref[i] == 0:
                    ref[i] = speaker_dict[spkname] 
                else:
                    ref[i] = 1 # The overlapping speech is marked as 1.
    ref = ref.astype(int)

    print("{} SPEAKERS IN {}".format(num_spk, uttname))
    print("{} TOTAL, {} SILENCE({:.0f}%), {} OVERLAPPING({:.0f}%)".format(len(ref), np.sum(ref == 0), 100.0 * np.sum(ref == 0) / len(ref), np.sum(ref == 1), 100.0 * np.sum(ref == 1) / len(ref)))

    duration_list = []
    for i in range(num_spk):
        duration_list.append(1.0 * np.sum(ref == (i + 2)) / len(ref))
    duration_list.sort()
    duration_list = map(lambda x: '{0:.2f}'.format(x), duration_list)
    print("DISTRIBUTION OF SPEAKER {}".format(" ".join(duration_list)))
    print("")
    sys.stdout.flush()
    utt_rttm_file.close()
    return ref

def create_rttm_output(uttname, predicted_label, output_dir, channel):
    num_frames = len(predicted_label)

    start_idx = 0
    idx_list = []

    last_label = predicted_label[0]
    for i in range(num_frames):
        if predicted_label[i] == last_label: # The speaker label remains the same.
            continue
        else: # The speaker label is different.
            if last_label != 0: # Ignore the silence.
                idx_list.append([start_idx, i, last_label])
            start_idx = i
            last_label = predicted_label[i]
    if last_label != 0:
        idx_list.append([start_idx, num_frames, last_label])

    with open("{}/{}_predict.rttm".format(output_dir, uttname), 'w') as fh:
        for i in range(len(idx_list)):
            start_frame = (idx_list[i])[0]
            end_frame = (idx_list[i])[1]
            label = (idx_list[i])[2]
            duration = end_frame - start_frame
            fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(uttname, channel, start_frame / 100.0, duration / 100.0, label))
    return 0

def match_DER(string):
    string_split = string.split('\n')
    for line in string_split:
        if "OVERALL SPEAKER DIARIZATION ERROR" in line:
            return line
    return 0

def main():
    parser = argparse.ArgumentParser(description='VB Resegmentation')
    parser.add_argument('data_dir', type=str, help='Subset data directory')
    parser.add_argument('init_rttm_filename', type=str, 
                        help='The rttm file to initialize the VB system, usually the AHC cluster result')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('dubm_model', type=str, help='Path of the diagonal UBM model')
    parser.add_argument('ie_model', type=str, help='Path of the ivector extractor model')

    parser.add_argument('--true-rttm-filename', type=str, default="None",
                        help='The true rttm label file')
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
    data_dir = args.data_dir
    init_rttm_filename = args.init_rttm_filename

    # The data directory should contain wav.scp, spk2utt, utt2spk and feats.scp
    utt2spk_filename = "{}/utt2spk".format(data_dir) 
    utt2num_frames_filename = "{}/utt2num_frames".format(data_dir) 
    feats_scp_filename = "{}/feats.scp".format(data_dir)
    temp_dir = "{}/tmp".format(args.output_dir)
    rttm_dir = "{}/rttm".format(args.output_dir)

    utt_list = get_utt_list(utt2spk_filename)
    utt2num_frames = utt_num_frames_mapping(utt2num_frames_filename) 
    print("------------------------------------------------------------------------")
    print("")
    sys.stdout.flush()
    
    # Load the diagonal UBM and i-vector extractor
    with open(args.dubm_model, 'rb') as fh:
        dubm_para = pickle.load(fh)
    with open(args.ie_model, 'rb') as fh:
        ie_para = pickle.load(fh)
    
    DUBM_WEIGHTS = None
    DUBM_MEANS_INVVARS = None
    DUBM_INV_VARS = None
    IE_M = None

    for key in dubm_para.keys():
        if key == "<WEIGHTS>":
            DUBM_WEIGHTS = dubm_para[key]
        elif key == "<MEANS_INVVARS>":
            DUBM_MEANS_INVVARS = dubm_para[key]
        elif key == "<INV_VARS>":
            DUBM_INV_VARS = dubm_para[key]
        else:
            continue
        
    for key in ie_para.keys():
        if key == "M":
            IE_M = np.transpose(ie_para[key], (2, 0, 1)) 
    m = DUBM_MEANS_INVVARS / DUBM_INV_VARS
    iE = DUBM_INV_VARS
    w = DUBM_WEIGHTS
    V = IE_M

    # Load the MFCC features
    feats_dict = {}
    for key,mat in kaldi_io.read_mat_scp(feats_scp_filename):
        feats_dict[key] = mat

    for utt in utt_list:
        # Get the alignments from the clustering result.
        # In init_ref, 0 denotes the silence silence frames
        # 1 denotes the overlapping speech frames, the speaker
        # label starts from 2.
        init_ref = create_ref_file(utt, utt2num_frames, init_rttm_filename, temp_dir, "{}.rttm".format(utt))
        # Ground truth of the diarization.
        if args.true_rttm_filename != "None":
            true_ref = create_ref_file(utt, utt2num_frames, args.true_rttm_filename, temp_dir, "{}_true.rttm".format(utt))
        else:
            true_ref = None

        X = feats_dict[utt]
        X = X.astype(np.float64)

        # Keep only the voiced frames (0 denotes the silence 
        # frames, 1 denotes the overlapping speech frames). Since
        # our method predicts single speaker label for each frame
        # the init_ref doesn't contain 1.
        mask = (init_ref >= 2)
        X_voiced = X[mask]
        init_ref_voiced = init_ref[mask] - 2

        if args.true_rttm_filename != "None": 
            true_ref_voiced = true_ref[mask] - 2
            if np.sum(true_ref) == 0:
                print("Warning: {} has no voiced frames in the label file".format(utt))
                continue
        if X_voiced.shape[0] == 0:
            print("Warning: {} has no voiced frames in the initialization file".format(utt))
            continue

        # Initialize the posterior of each speaker based on the clustering result.
        if args.initialize:
            q = VB_diarization.frame_labels2posterior_mx(init_ref_voiced, args.max_speakers)
            if args.true_rttm_filename != "None": 
                cmd = "md-eval.pl -1 -c 0.25 -r {}/{}_true.rttm -s {}/{}.rttm 2".format(temp_dir, utt, temp_dir, utt)
                status, output = commands.getstatusoutput(cmd) 
                assert status == 0
                DER_info = match_DER(output)
                print("BEFORE RUNNING VB RESEGMENTATION")
                print(DER_info + "\n")
        else:
            q = None
            print("RANDOM INITIALIZATION\n")
        
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

        duration_list = []
        for i in range(args.max_speakers):
            num_frames = np.sum(predicted_label == (i + 2))
            if num_frames == 0:
                continue
            else:
                duration_list.append(1.0 * num_frames / len(predicted_label))
        duration_list.sort()
        duration_list = map(lambda x: '{0:.2f}'.format(x), duration_list)
        print("PREDICTED {} SPEAKERS".format(len(duration_list)))
        print("DISTRIBUTION {}".format(" ".join(duration_list)))
        print("sp_out", sp_out)
        print("L_out", L_out)

        # Create the output rttm file and compute the DER after re-segmentation
        create_rttm_output(utt, predicted_label, rttm_dir, args.channel)
        if args.true_rttm_filename != "None":
            cmd = "md-eval.pl -1 -c 0.25 -r {}/{}_true.rttm -s {}/{}_predict.rttm 2".format(temp_dir, utt, rttm_dir, utt)
            status, output = commands.getstatusoutput(cmd) 
            assert status == 0
            DER_info = match_DER(output)
            print("")
            print("AFTER RUNNING VB RESEGMENTATION")
            print(DER_info)
        print("")
        print("------------------------------------------------------------------------")
        print("")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    main()
