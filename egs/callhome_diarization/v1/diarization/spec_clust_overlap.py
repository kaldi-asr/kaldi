# Copyright  2020  Maxim Korenevsky (STC-innovations Ltd)
#            2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

import argparse, os, re, sys
import numpy as np
from kaldiio import ReadHelper, WriteHelper
import scipy
from sklearn.cluster import SpectralClustering

'''
    This script is similar to spec_clust.py, with the difference that a modified form of the
    "discretize" method is used to cluster (after deciding the number of clusters). The
    modification allows to use an additional input containing the overlapping speech segments,
    and finally labels are allocated possibly containing multiple speakers in each segment.
'''

#   Input-output routines

def LoadAffinityMatrix(file):
    Matrices=dict()
    with ReadHelper(file) as reader:
        for key, np_arr in reader:
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
                label = labels[id][i]
                if type(label) is tuple:
                    f.write('{} {}\n'.format(IDs[id][i],label[0]+1))
                    f.write('{} {}\n'.format(IDs[id][i],label[1]+1))
                else:
                    f.write('{} {}\n'.format(IDs[id][i],label+1))

def GetOverlapDecision(overlap_segs, subsegment, frac = 0.5):
    """ Returns true if at least 'frac' fraction of the subsegment lies
    in the overlap_segs."""
    start_time = subsegment[0]
    end_time = subsegment[1]
    dur = end_time - start_time
    total_ovl = 0
    
    for seg in enumerate(overlap_segs):
        cur_start, cur_end = seg
        if (cur_start >= end_time):
            break
        ovl_start = max(start_time, cur_start)
        ovl_end = min(end_time, cur_end)
        ovl_time = max(0, ovl_end-ovl_start)

        total_ovl += ovl_time
    
    return (total_ovl >= frac * dur)


def ComputeOverlapVector(IDs, overlap_rttm):
    overlap_segs = []
    with open(overlap_rttm, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            overlap_segs.append((parts[1], float(parts[3]), float(parts[3]) + float(parts[4])))
    overlap_vectors = {}
    for id in IDs:
        ol_vec = np.zeros(len(IDs[id]))
        ol_segs = []
        for seg in overlap_segs:
            if (id == seg[0]):
                ol_segs.append((seg[1], seg[2]))
        ol_segs.sort(key=lambda x: x[0])
        if ol_segs is None:
            overlap_vectors[id] = ol_vec
            continue
        for i, segment in enumerate(IDs[id]):
            parts = re.split('_|-',segment)
            start_time = (float(parts[3]) + float(parts[5]))/100
            end_time = (float(parts[3]) + float(parts[6]))/100

            is_overlap = GetOverlapDecision(ol_segs, (start_time, end_time))
            if is_overlap:
                ol_vec[i] = 1
        print ("{}: {} fraction of segments are overlapping".format(id, ol_vec.sum()/len(ol_vec)), end='')
        overlap_vectors[id] = ol_vec
    return overlap_vectors



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
def NME_SpectralClustering(A, OLVec, num_clusters = None, max_num_clusters = 10, pbest = 0, pmax = 20):
    if pbest==0:
        print('Selecting best number of neighbors for affinity matrix thresolding:')
        rbest = None
        kbest = None
        for p in range(2, pmax+1):
            e, g, k, r = ComputeNMEParameters(A, p, max_num_clusters)
            print('p={}, r={}'.format(p,r))
            if rbest is None or rbest > r:
                rbest = r
                pbest = p
                kbest = k
        print('Best number of neighbors is {}'.format(pbest))
        return NME_SpectralClustering_sklearn(A, OLVec, num_clusters if num_clusters is not None else (kbest+1), pbest)
    if num_clusters is None:
        print('Compute number of clusters to generate:')
        e, g, r, k = ComputeNMEParameters(A, p, max_num_clusters)
        print('Number of clusters to generate is {}'.format(k+1))
        return NME_SpectralClustering_sklearn(A, OLVec, k+1, pbest)
    return NME_SpectralClustering_sklearn(A, OLVec, num_clusters, pbest)

'''
Performs spectral clustering with Normalized Maximum Eigengap (NME) with fixed threshold and number of clusters
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   OLVec: 0/1 vector denoting which segments are overlap segments
   num_clusters: number of clusters to generate
   pbest: best count for matrix binarization
Returns: cluster assignments for every speaker embedding   
'''
def NME_SpectralClustering_sklearn(A, OLVec, num_clusters, pbest):
    Ap = Threshold(A, pbest)
    Ap = (Ap + np.transpose(Ap)) / 2
    model = SpectralClustering(n_clusters = num_clusters, affinity='precomputed', random_state=0, 
        assign_labels='discretize_ol', overlap_vector=OLVec)
    labels = model.fit_predict(Ap)
    print (labels)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: spec_clust.py [options] <scores-rspec> <reco2utt-rspec> <labels-wspec>\n' +
                                                 'Performs spectral clustering of xvectors according to pairwise similarity scores\n' +
                                                 'Auto-selects binarization threshold')
    parser.add_argument('simmat_rspec', type=str, help='Kaldi-style rspecifier of similarity scores matrices to read')
    parser.add_argument('reco2utt_rspec', type=str, help='Kaldi-style rspecifier of recording-to-utterances correspondence')
    parser.add_argument('labels_wspec', type=str, help='Kaldi-style wspecifier to save xvector cluster labels')
    parser.add_argument('--max_neighbors', type=int, default=20, help='Maximum number of neighbors to threshold similarity matrix')
    parser.add_argument('--reco2num_spk', type=str, default='', help='Kaldi-style rspecifier of recording-to-numofspeakers correspondence')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters to generate. Ignored if --reco2num_spk is given')
    parser.add_argument('--overlap_rttm', type=str, default=None, help='Path to RTTM file containing overlap segments')
    args = parser.parse_args()

    assert args.max_neighbors > 1, 'Maximum number of neighbors should be at least 2, {} passed\n'.format(args.max_neighbors)

    print('Spectral clustering of xvector according to precomputed similarity scores matrix')
    print('Parameters:')
    print('Similarity matrix rspecifier: {}'.format(args.simmat_rspec))
    print('Reco2Utt rspecifier: {}'.format(args.reco2utt_rspec))
    print('Labels wspecifier: {}'.format(args.labels_wspec))
    print('Number of clusters to generate: {}'.format(args.num_clusters))
    print('Maximum number of nighbors to threshold similarity matrix: {}\n'.format(args.max_neighbors))
    print('Reco2NumSpk rspecifier: {}'.format(args.reco2num_spk))
    print('Overlap RTTM: {}'.format(args.overlap_rttm))

    print('Loading affinity matrices...', end='')
    Matrices = LoadAffinityMatrix(args.simmat_rspec)
    print('done')
    print('Loading Reco2Utt correspondence...', end='')
    IDs = LoadReco2Utt(args.reco2utt_rspec)
    print('done')

    if args.reco2num_spk != '':
        NumSpk = LoadReco2NumSpk(args.reco2num_spk)

    print('Getting overlap segments...')
    Overlaps = ComputeOverlapVector(IDs, args.overlap_rttm)

    Labels = dict()
    for id in IDs:
        A = Matrices[id]
        IDList = IDs[id]
        OLVec = Overlaps[id]

        num_clusters = args.num_clusters if args.reco2num_spk == '' else NumSpk[id]
        assert num_clusters is None or num_clusters > 0, 'Positive number of clusters expected for {}, {} found\n'.format(id, num_clusters)

        print('Start clustering for recording {}...'.format(id))
        Labels[id] = NME_SpectralClustering(A, OLVec, num_clusters = num_clusters, pmax = args.max_neighbors)
        print('Clustering done')
    print( 'Saving labels...')
    SaveLabels(IDs, Labels, args.labels_wspec)
    print('done')