#!/usr/bin/env python
# Copyright 2020 Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# This script is based on the Bayesian HMM-based xvector clustering
# code released by BUTSpeech at: https://github.com/BUTSpeechFIT/VBx.
# Note that this assumes that the provided labels are for a single
# recording. So this should be called from a script such as
# vb_hmm_xvector.sh which can divide all labels into per recording
# labels.

import sys, argparse, struct, re
import numpy as np
import itertools
import kaldi_io

from scipy.special import softmax

import VB_diarization

########### HELPER FUNCTIONS #####################################

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script performs Bayesian HMM-based
            clustering of x-vectors for one recording""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--init-smoothing", type=float, default=10,
        help="AHC produces hard assignments of x-vetors to speakers."
        " These are smoothed to soft assignments as the initialization"
        " for VB-HMM. This parameter controls the amount of smoothing."
        " Not so important, high value (e.g. 10) is OK  => keeping hard assigment")
    parser.add_argument("--loop-prob", type=float, default=0.80,
                        help="probability of not switching speakers between frames")
    parser.add_argument("--fa", type=float, default=0.4,
                        help="scale sufficient statistics collected using UBM")
    parser.add_argument("--fb", type=float, default=11,
                        help="speaker regularization coefficient Fb (controls final # of speaker)")
    parser.add_argument("--overlap-rttm", type=str,
                        help="path to an RTTM file containing overlap segments. If provided,"
                        "multiple speaker labels will be allocated to these segments.")
    parser.add_argument("xvector_ark_file", type=str,
                        help="Ark file containing xvectors for all subsegments")
    parser.add_argument("plda", type=str,
                        help="path to PLDA model")
    parser.add_argument("input_label_file", type=str,
                        help="path of input label file")
    parser.add_argument("output_label_file", type=str,
                        help="path of output label file")
    args = parser.parse_args()
    return args

def read_labels_file(label_file):
    segments = []
    labels = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            segment, label = line.strip().split()
            segments.append(segment)
            labels.append(int(label))
    return segments, labels

def write_labels_file(seg2label, out_file):
    f = open(out_file, 'w')
    for seg in sorted(seg2label.keys()):
        label = seg2label[seg]
        if type(label) is tuple:
            f.write("{} {}\n".format(seg, label[0]))
            f.write("{} {}\n".format(seg, label[1]))
        else:
            f.write("{} {}\n".format(seg, label))
    f.close()
    return

def get_overlap_decision(overlap_segs, subsegment, frac = 0.5):
    """ Returns true if at least 'frac' fraction of the subsegment lies
    in the overlap_segs."""
    start_time = subsegment[0]
    end_time = subsegment[1]
    dur = end_time - start_time
    total_ovl = 0
    
    for seg in overlap_segs:
        cur_start, cur_end = seg
        if (cur_start >= end_time):
            break
        ovl_start = max(start_time, cur_start)
        ovl_end = min(end_time, cur_end)
        ovl_time = max(0, ovl_end-ovl_start)

        total_ovl += ovl_time
    
    return (total_ovl >= frac * dur)


def get_overlap_vector(overlap_rttm, segments):
    reco_id = '_'.join(segments[0].split('_')[:3])
    overlap_segs = []
    with open(overlap_rttm, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if (parts[1] == reco_id):
                overlap_segs.append((float(parts[3]), float(parts[3]) + float(parts[4])))
    ol_vec = np.zeros(len(segments))
    overlap_segs.sort(key=lambda x: x[0])
    for i, segment in enumerate(segments):
        parts = re.split('_|-',segment)
        start_time = (float(parts[3]) + float(parts[5]))/100
        end_time = (float(parts[3]) + float(parts[6]))/100

        is_overlap = get_overlap_decision(overlap_segs, (start_time, end_time))
        if is_overlap:
            ol_vec[i] = 1
    print ("{}: {} fraction of segments are overlapping".format(id, ol_vec.sum()/len(ol_vec)))
    return ol_vec

def read_args(args):
    segments, labels = read_labels_file(args.input_label_file)
    xvec_all = dict(kaldi_io.read_vec_flt_ark(args.xvector_ark_file))
    xvectors = []
    for segment in segments:
        xvectors.append(xvec_all[segment])
    _, _, plda_psi = kaldi_io.read_plda(args.plda)
    if (args.overlap_rttm is not None):
        print('Getting overlap segments...')
        overlaps = get_overlap_vector(args.overlap_rttm, segments)
    else:
        overlaps = None
    return xvectors, segments, labels, plda_psi, overlaps


###################################################################

def vb_hmm(segments, in_labels, xvectors, overlaps, plda_psi, init_smoothing, loop_prob, fa, fb):
    x = np.array(xvectors)
    dim = x.shape[1]

    # Smooth the hard labels obtained from AHC to soft assignments of x-vectors to speakers
    q_init = np.zeros((len(in_labels), np.max(in_labels)+1))
    q_init[range(len(in_labels)), in_labels] = 1.0
    q_init = softmax(q_init*init_smoothing, axis=1)

    # Prepare model for VB-HMM clustering
    ubmWeights = np.array([1.0])
    ubmMeans = np.zeros((1,dim))
    invSigma= np.ones((1,dim))
    V=np.diag(np.sqrt(plda_psi[:dim]))[:,np.newaxis,:]

    # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
    # => GMM with only 1 component, V derived across-class covariance, and invSigma is inverse 
    # within-class covariance (i.e. identity)
    q, _, _ = VB_diarization.VB_diarization(x, ubmMeans, invSigma, ubmWeights, V, pi=None, 
        gamma=q_init, maxSpeakers=q_init.shape[1], maxIters=40, epsilon=1e-6, loopProb=loop_prob,
        Fa=fa, Fb=fb)

    labels = np.argsort(q, axis=1)[:,[-1,-2]]

    if overlaps is not None:
        final_labels = []
        for i in range(len(overlaps)):
            if (overlaps[i] == 1):
                final_labels.append((labels[i,0], labels[i,1]))
            else:
                final_labels.append(labels[i,0])
    else:
        final_labels = labels[:,0]
    
    return {seg:label for seg,label in zip(segments,final_labels)}

def main():
    args = get_args()
    xvectors, segments, labels, plda_psi, overlaps = read_args(args)

    seg2label_vb = vb_hmm(segments, labels, xvectors, overlaps, plda_psi, args.init_smoothing, 
        args.loop_prob, args.fa, args.fb)
    write_labels_file(seg2label_vb, args.output_label_file)

if __name__=="__main__":
    main()

