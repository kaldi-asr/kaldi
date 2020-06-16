#!/usr/bin/env python3

# Copyright  2020  Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

import argparse
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform
from kaldiio import ReadHelper, WriteHelper


def LoadReco2Utt(file):
    if ':' in file:
        file = file.split(':')[1]
    IDs=dict()
    with open(file,'r') as f:
        for line in f:
            ids = line.strip().split()
            IDs[ids[0]] = ids[1:]
    return IDs

def ReadXvecs(rspec):
    xvecs=dict()
    with ReadHelper(rspec) as reader:
        for utid, xvec in reader:
            xvecs[utid] = xvec
    reader.close()
    return xvecs

def Normalize(xvecs_in):
    N = len(xvecs_in)
    xvec_mean=np.zeros(xvecs_in[0].shape)
    for i in range(N):
        xvec_mean += xvecs_in[i]
    xvec_mean /= N
    xvecs = np.copy(xvecs_in)
    for i in range(N):
        xvecs[i] -= xvec_mean
        xvecs[i] = xvecs[i] / np.linalg.norm(xvecs[i])
    return xvecs

def CalcCosSim(vecs):
    return 1 - squareform(pdist(np.asarray(vecs), 'cosine'))

def WriteDistMatrices(D, wspec):
    with WriteHelper(wspec) as writer:
        for id in D:
            writer(id, D[id])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: calc_cossim_scores.py <reco2utt-rspec> <xvec-rspec> <simmat-wspec>\nComputes matrices of the cosine similarity scores between normalized x-vectors for each recording')
    parser.add_argument('reco2utt', type=str, help='Kaldi-style rspecifier of recording to segments correspondence')
    parser.add_argument('xvec_rspec', type=str, help='Kaldi-style rspecifier of segment xvectors to read')
    parser.add_argument('simmat_wspec', type=str, help='Kaldi-style wspecifier of similarity matrices to write')
    args = parser.parse_args()


    print('Computing cosine similarity matrix between ivectors')
    print('Parameters:')
    print('Reco2Utt rspecifier: {}'.format(args.reco2utt))
    print('Xvectors rspecifier: {}'.format(args.xvec_rspec))
    print('Similarity matrices wspecifier: {}'.format(args.simmat_wspec))

    IDs = LoadReco2Utt(args.reco2utt)
    xvecs_all = ReadXvecs(args.xvec_rspec)
    D = dict()
    for reco_id in IDs:
        xvecs = [ xvecs_all[id] for id in IDs[reco_id] ]
        xvecs = Normalize(xvecs)                              # !!!! Normalize per recording (session) !!!!
        D[reco_id] = CalcCosSim(xvecs)
    WriteDistMatrices(D, args.simmat_wspec)