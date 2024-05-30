#!/usr/bin/env python3

# Copyright  2020  Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

import argparse
import os
import numpy as np
from sklearn.cluster import k_means
import kaldi_io
import scipy
from sklearn.cluster import SpectralClustering

'''
   Spectral Clustering based on binarization and automatic thresholding
   Paper: T.Park, K.Han, M.Kumar, and S.Narayanan, Auto-tuning spectral clustering for speaker diarization using normalized maximumeigengap, IEEE Signal Processing Letters, vol. 27, pp. 381-385,2019

 This version has a few differences from the original algorithm (thanks Naoyuki Kanda for pointing this out):
 1. The original paper performs clustering on a binarized affinity matrix. Here, a thresholded (but not
 binarized) matrix is used.
 2. The original paper uses an unnormalized Laplacian for the clustering. Here, scikit-learn's version is used
 which performs normalization of the Laplacian.
'''

#   Input-output routines

def LoadAffinityMatrix(file):
    Matrices=dict()
    for key, np_arr in kaldi_io.read_mat_scp(file):
        Matrices[key] = np_arr
    return Matrices

def LoadReco2Utt(file):
    if ':' in file:
        file = file.split(':')[1]
    IDs=dict()
    with open(file,'r') as f:
        for line in f:
            ids = line.strip().split()
            IDs[ids[0]] = ids[1:]
    return IDs


def LoadReco2NumSpk(file):
    if ':' in file:
        file = file.split(':')[1]
    NumSpk=dict()
    with open(file,'r') as f:
        for line in f:
            ids = line.strip().split()
            NumSpk[ids[0]] = int(ids[1])
    return NumSpk

def SaveLabels(IDs, labels, file):
    if ':' in file:
        file = file.split(':')[1]
    with open(file,'w') as f:
        for id in IDs:
            for i in range(len(IDs[id])):
                f.write('{} {}\n'.format(IDs[id][i], labels[id][i]+1))

#   NME low-level operations

# Prepares binarized(0/1) affinity matrix with p_neighbors non-zero elements in each row
def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out

# Thresolds affinity matrix to leave p maximum non-zero elements in each row
def Threshold(A, p):
    N = A.shape[0]
    Ap = np.zeros((N,N))
    for i in range(N):
        thr = sorted(A[i,:], reverse=True)[p]
        Ap[i,A[i,:]>thr] = A[i,A[i,:]>thr]
    return Ap

# Computes Laplacian of a matrix
def Laplacian(A):
    d = np.sum(A, axis=1)-np.diag(A)
    D = np.diag(d)
    return D - A

# Calculates eigengaps (differences between adjacent eigenvalues sorted in descending order)
def Eigengap(S):
    S = sorted(S)
    return np.diff(S)

# Computes parameters of normalized eigenmaps for automatic thresholding selection
def ComputeNMEParameters(A, p, max_num_clusters):
    # p-Neighbour binarization
    Ap = get_kneighbors_conn(A, p)
    # Symmetrization
    Ap = (Ap + np.transpose(Ap))/2
    # Laplacian matrix computation
    Lp = Laplacian(Ap)
    # EigenValue Decomposition
    S, eig_vecs = scipy.linalg.eigh(Lp)
    # Eigengap computation
    e = Eigengap(S)
    g = np.max(e[:max_num_clusters])/(np.max(S)+1e-10)
    N = Lp.shape[0]
    # EigenValue Decomposition
    # Get max_num_clusters+1 lowest eigenvalues sorted in ascending order
    S, _ = scipy.sparse.linalg.eigsh(-Lp, k=max_num_clusters+1, which='LA')
    S = -S[::-1]
    # Get largest eigenvalue
    Smax, _ = scipy.sparse.linalg.eigsh(Lp, k=1, which='LA')
    # Eigengap computation
    e = Eigengap(S)
    # g = np.max(e[:max_num_clusters])/(np.max(S)+1e-10)
    g = np.max(e[:max_num_clusters])/(Smax[0]+1e-10)
    r = p/g
    k = np.argmax(e[:max_num_clusters])
    return (e, g, k, r)


'''
Performs spectral clustering with Normalized Maximum Eigengap (NME)
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   num_clusters: number of clusters to generate (if None, determined automatically)
   max_num_clusters: maximum allowed number of clusters to generate
   pmax: maximum count for matrix binarization (should be at least 2)
   pbest: best count for matrix binarization (if 0, determined automatically)
Returns: cluster assignments for every speaker embedding   
'''
def NME_SpectralClustering(A, num_clusters = None, max_num_clusters = 10, pbest = 0, pmin = 2, pmax = 20):
    # First estimate number of clusters by scanning different thresholds for
    # affinity matrix
    if pbest==0:
        print('Selecting best number of neighbors for affinity matrix thresolding:')
        rbest = None
        kbest = None
        for p in range(pmin, pmax+1):
            _, _, k, r = ComputeNMEParameters(A, p, max_num_clusters)
            print('p={}, r={}'.format(p,r))
            if rbest is None or rbest > r:
                rbest = r
                pbest = p
                kbest = k
        print('Best number of neighbors is {}'.format(pbest))
    else:
        _, _, kbest, _ = ComputeNMEParameters(A, pbest, max_num_clusters)
    
    num_clusters = num_clusters if num_clusters is not None else kbest+1
    print('Number of clusters: {}'.format(num_clusters))
    return NME_SpectralClustering_sklearn(A, num_clusters, pbest)

'''
Performs spectral clustering with Normalized Maximum Eigengap (NME) with fixed threshold and number of clusters
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   num_clusters: number of clusters to generate
   pbest: best count for matrix binarization
Returns: cluster assignments for every speaker embedding   
'''
def NME_SpectralClustering_sklearn(A, num_clusters, pbest):
    Ap = Threshold(A, pbest)
    Ap = (Ap + np.transpose(Ap)) / 2
    model = SpectralClustering(n_clusters = num_clusters, affinity='precomputed', random_state=0)
    labels = model.fit_predict(Ap)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: spec_clust.py [options] <scores-rspec> <reco2utt-rspec> <labels-wspec>\n' +
                                                 'Performs spectral clustering of xvectors according to pairwise similarity scores\n' +
                                                 'Auto-selects binarization threshold')
    parser.add_argument('simmat_rspec', type=str, help='Kaldi-style rspecifier of similarity scores matrices to read')
    parser.add_argument('reco2utt_rspec', type=str, help='Kaldi-style rspecifier of recording-to-utterances correspondence')
    parser.add_argument('labels_wspec', type=str, help='Kaldi-style wspecifier to save xvector cluster labels')
    parser.add_argument('--min_neighbors', type=int, default=2, help='Minimum number of neighbors to threshold similarity matrix')
    parser.add_argument('--max_neighbors', type=int, default=20, help='Maximum number of neighbors to threshold similarity matrix')
    parser.add_argument('--reco2num_spk', type=str, default='', help='Kaldi-style rspecifier of recording-to-numofspeakers correspondence')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters to generate. Ignored if --reco2num_spk is given')
    args = parser.parse_args()

    assert args.max_neighbors > 1, 'Maximum number of neighpors should be at least 2, {} passed\n'.format(args.max_neighbors)

    print('Spectral clustering of xvector according to precomputed similarity scores matrix')
    print('Parameters:')
    print('Similarity matrix rspecifier: {}'.format(args.simmat_rspec))
    print('Reco2Utt rspecifier: {}'.format(args.reco2utt_rspec))
    print('Labels wspecifier: {}'.format(args.labels_wspec))
    print('Number of clusters to generate: {}'.format(args.num_clusters))
    print('Number of neighbors to threshold similarity matrix: {}-{}\n'.format(args.min_neighbors, args.max_neighbors))
    print('Reco2NumSpk rspecifier: {}'.format(args.reco2num_spk))

    print('Loading affinity matrices...', end='')
    Matrices = LoadAffinityMatrix(args.simmat_rspec)
    print('done')
    print('Loading Reco2Utt correspondence...', end='')
    IDs = LoadReco2Utt(args.reco2utt_rspec)
    print('done')

    if args.reco2num_spk != '':
        NumSpk = LoadReco2NumSpk(args.reco2num_spk)

    Labels = dict()
    for id in IDs:
        A = Matrices[id]
        IDList = IDs[id]

        num_clusters = args.num_clusters if args.reco2num_spk == '' else NumSpk[id]
        assert num_clusters is None or num_clusters > 0, 'Positive number of clusters expected for {}, {} found\n'.format(id, num_clusters)

        print('Start clustering for recording {}...'.format(id))
        Labels[id] = NME_SpectralClustering(A, num_clusters = num_clusters, pmin = args.min_neighbors, pmax = args.max_neighbors)
        print('Clustering done')
    print( 'Saving labels...')
    SaveLabels(IDs, Labels, args.labels_wspec)
    print('done')
