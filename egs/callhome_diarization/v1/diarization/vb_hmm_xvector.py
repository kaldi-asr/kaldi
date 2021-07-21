#!/usr/bin/env python3
# Copyright 2020 Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# This script is based on the Bayesian HMM-based xvector clustering
# code released by BUTSpeech at: https://github.com/BUTSpeechFIT/VBx.
# Note that this assumes that the provided labels are for a single
# recording. So this should be called from a script such as
# vb_hmm_xvector.sh which can divide all labels into per recording
# labels.

import sys
import argparse
import struct
import itertools

import numpy as np
import kaldi_io
from scipy.special import softmax

sys.path.insert(0, 'steps')
import libs.common as common_lib
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
    parser.add_argument("xvector_scp_file", type=str,
                        help="Scp file containing xvectors for all subsegments")
    parser.add_argument("reco2segs", type=str, help="Spk2utt style file "
                        "containing recording to segment information")
    parser.add_argument("plda", type=str,
                        help="path to PLDA model")
    parser.add_argument("input_label_file", type=str,
                        help="path of input label file")
    parser.add_argument("output_label_file", type=str,
                        help="path of output label file")
    args = parser.parse_args()
    return args

def read_labels_file(label_file):
    seg2label = {}
    with common_lib.smart_open(label_file, 'r') as f:
        for line in f:
            segment, label = line.strip().split(" ")
            seg2label[segment]  = int(label)
    return seg2label

def read_reco2seg_file(reco2seg_file):
    reco2segs = {}
    with common_lib.smart_open(reco2seg_file, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            reco2segs[parts[0]] = parts[1:]
    return reco2segs

def write_labels_file(seg2label, out_file):
    with common_lib.smart_open(out_file, 'w') as f:
        for seg in sorted(seg2label.keys()):
            label = seg2label[seg]
            f.write("{0} {1}\n".format(seg, label))

def read_args(args):
    seg2label = read_labels_file(args.input_label_file)
    reco2segs = read_reco2seg_file(args.reco2segs)
    xvectors = dict(kaldi_io.read_vec_flt_scp(args.xvector_scp_file))
    _, _, plda_psi = kaldi_io.read_plda(args.plda)
    return xvectors, seg2label, reco2segs, plda_psi


###################################################################

def vb_hmm(segments, in_labels, xvectors, plda_psi, init_smoothing, loop_prob, fa, fb):
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

    labels = np.unique(q.argmax(1), return_inverse=True)[1] 

    return {seg:label for seg,label in zip(segments,labels)}

def main():
    args = get_args()
    xvectors_all, seg2label, reco2segs, plda_psi = read_args(args)

    seg2label_vb = {}
    for reco_id in reco2segs:
        segments = reco2segs[reco_id]
        labels = [seg2label[seg] for seg in segments]
        xvectors = [xvectors_all[k] for k in segments]
        seg2label_vb.update(
            vb_hmm(
                segments, labels, xvectors, plda_psi, args.init_smoothing,
                args.loop_prob, args.fa, args.fb
        ))
    write_labels_file(seg2label_vb, args.output_label_file)

if __name__=="__main__":
    main()
