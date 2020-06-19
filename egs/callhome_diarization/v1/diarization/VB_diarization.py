#!/usr/bin/env python3
# Copyright 2013-2017 Lukas Burget (burget@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Revision History
#   L. Burget   16/07/13 01:00AM - original version
#   L. Burget   20/06/17 12:07AM - np.asarray replaced by .toarray()
#                                - minor bug fix in initializing q
#                                - minor bug fix in ELBO calculation
#                                - few more optimizations

import numpy as np
from scipy.sparse import coo_matrix
import scipy.linalg as spl
#import numexpr as ne # the dependency on this modul can be avoided by replacing
#                       # logsumexp_ne and exp_ne with logsumexp and np.exp

#[q sp Li] =
def VB_diarization(X, m, iE, w, V, sp=None, q=None,
                   maxSpeakers = 10, maxIters = 10,
                   epsilon = 1e-4, loopProb = 0.99, statScale = 1.0,
                   alphaQInit = 1.0, downsample = None, VtiEV = None, ref=None,
                   plot=False, sparsityThr=0.001, llScale=1.0, minDur=1):

  """
  This a generalized version of speaker diarization described in:

  Kenny, P. Bayesian Analysis of Speaker Diarization with Eigenvoice Priors,
  Montreal, CRIM, May 2008.

  Kenny, P., Reynolds, D., and Castaldo, F. Diarization of Telephone
  Conversations using Factor Analysis IEEE Journal of Selected Topics in Signal
  Processing, December 2010.

  The generalization introduced in this implementation lies in using an HMM
  instead of the simple mixture model when modeling generation of segments
  (or even frames) from speakers. HMM limits the probability of switching
  between speakers when changing frames, which makes it possible to use
  the model on frame-by-frame bases without any need to iterate between
  1) clustering speech segments and 2) re-segmentation (i.e. as it was done in
  the paper above).

  Inputs:
  X  - T x D array, where columns are D dimensional feature vectors for T frames
  m  - C x D array of GMM component means
  iE - C x D array of GMM component inverse covariance matrix diagonals
  w  - C dimensional column vector of GMM component weights
  V  - R x C x D array of eigenvoices
  maxSpeakers - maximum number of speakers expected in the utterance
  maxIters    - maximum number of algorithm iterations
  epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
  loopProb    - probability of not switching speakers between frames
  statScale   - scale sufficient statiscits collected using UBM
  llScale     - scale UBM likelihood (i.e. llScale < 1.0 make atribution of
                frames to UBM componets more uncertain)
  sparsityThr - set occupations smaller that this threshold to 0.0 (saves memory
                as the posteriors are represented by sparse matrix)
  alphaQInit  - Dirichlet concentraion parameter for initializing q
  downsample  - perform diarization on input downsampled by this factor
  VtiEV       - C x (R**2+R)/2 matrix normally calculated by VB_diarization when
                VtiEV is None. However, it can be pre-calculated using function
                precalculate_VtiEV(V) and used across calls of VB_diarization.
  minDur      - minimum number of frames between speaker turns imposed by linear
                chains of HMM states corresponding to each speaker. All the states
                in a chain share the same output distribution
  ref         - T dim. integer vector with reference speaker ID (0:maxSpeakers)
                per frame
  plot        - if set to True, plot per-frame speaker posteriors.

   Outputs:
   q  - S x T matrix of posteriors attribution each frame to one of S possible
        speakers, where S is given by opts.maxSpeakers
   sp - S dimensional column vector of ML learned speaker priors. Ideally, these
        should allow to estimate # of speaker in the utterance as the
        probabilities of the redundant speaker should converge to zero.
   Li - values of auxiliary function (and DER and frame cross-entropy between q
        and reference if 'ref' is provided) over iterations.
  """

  # The references to equations corresponds to the technical report:
  # Kenny, P. Bayesian Analysis of Speaker Diarization with Eigenvoice Priors,
  # Montreal, CRIM, May 2008.

  D=X.shape[1]  # feature dimensionality
  C=len(w)      # number of mixture components
  R=V.shape[0]  # subspace rank
  nframes=X.shape[0]

  if VtiEV is None:
    VtiEV = precalculate_VtiEV(V, iE)

  V = V.reshape(V.shape[0],-1)

  if sp is None:
    sp = np.ones(maxSpeakers)/maxSpeakers
  else:
    maxSpeakers = len(sp)

  if q is None:
    # initialize q from flat Dirichlet prior with concentrsaion parameter alphaQInit
    q = np.random.gamma(alphaQInit, size=(nframes, maxSpeakers))
    q = q / q.sum(1, keepdims=True)

  # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics)
  ll = (X**2).dot(-0.5*iE.T) + X.dot(iE.T*m.T)-0.5*((iE * m**2 - np.log(iE)).sum(1) - 2*np.log(w) + D*np.log(2*np.pi))
  ll *= llScale
  G = logsumexp(ll, axis=1)
  NN =  np.exp(ll - G[:,np.newaxis]) * statScale
  NN[NN<sparsityThr] = 0.0

  #Kx = np.sum(NN * (np.log(w) - np.log(NN)), 1)
  NN = coo_matrix(NN) # represent zero-order stats using sparse matrix
  print('Sparsity: ', len(NN.row), float(len(NN.row))/np.prod(NN.shape))
  LL = np.sum(G) # total log-likelihod as calculated using UBM

  mixture_sum = coo_matrix((np.ones(C*D), (np.repeat(range(C),D), range(C*D))), shape=(C, C*D))

  #G = np.sum((NN.multiply(ll - np.log(w))).toarray(), 1) + Kx  # eq. (15) # Aleready calculated above

  # Calculate per-frame first order statistics projected into the R-dim. subspace
  # V^T \Sigma^{-1} F_m
  F_s = coo_matrix((((X[NN.row]-m[NN.col])*NN.data[:,np.newaxis]).flat,
                   (NN.row.repeat(D), NN.col.repeat(D)*D+np.tile(range(D), len(NN.col)))), shape=(nframes, D*C))
  VtiEF = F_s.tocsr().dot((iE.flat * V).T) ; del F_s
  ## The code above is only efficient implementation of the following comented code
  #VtiEF = 0;
  #for ii in range(C):
  #  VtiEF = VtiEF + V[ii*D:(ii+1)*D,:].T.dot(NN[ii,:] * np.sqrt(iE[:,[ii]]) *  (X - m[:,[ii]]))

  if downsample is not None:
    # Downsample NN, VtiEF, G and q by summing the statistic over 'downsample' frames
    # This speeds-up diarization for the price of lowering its frame resolution
    downsampler = coo_matrix((np.ones(nframes, dtype=np.int64), ((np.ceil(np.arange(nframes)/downsample)).astype(int), np.arange(nframes))), shape=(int(np.ceil((nframes - 1.0) / downsample)) + 1, nframes))
    NN    = downsampler.dot(NN)
    VtiEF = downsampler.dot(VtiEF)
    G     = downsampler.dot(G)
    q     = downsampler.dot(q) / downsample
  else:
    downsampler=np.array(1)

  Li = [[LL]] # for the 0-th iteration,
  if ref is not None:
    Li[-1] += [DER(downsampler.T.dot(q), ref), DER(downsampler.T.dot(q), ref, xentropy=True)]

  lls = np.zeros_like(q)
  tr = np.eye(minDur*maxSpeakers, k=1)
  ip = np.zeros(minDur*maxSpeakers)
  for ii in range(maxIters):
    L = 0 # objective function (37) (i.e. VB lower-bound on the evidence)
    Ns =   NN.T.dot(q).T                             # bracket in eq. (34) for all 's'
    VtNsiEV_flat = Ns.astype(VtiEV.dtype).dot(VtiEV) # eq. (34) except for 'I' for all 's'
    VtiEFs = q.T.dot(VtiEF)                          # eq. (35) except for \Lambda_s^{-1} for all 's'
    for sid in range(maxSpeakers):
        invL = np.linalg.inv(np.eye(R) + tril_to_sym(VtNsiEV_flat[sid])) # eq. (34) inverse
        a = invL.dot(VtiEFs[sid])                                        # eq. (35)
        # eq. (29) except for the prior term \ln \pi_s. Our prior is given by HMM
        # trasition probability matrix. Instead of eq. (30), we need to use
        # forward-backwar algorithm to calculate per-frame speaker posteriors,
        # where 'lls' plays role of HMM output log-probabilities
        lls[:,sid] = G + VtiEF.dot(a) - 0.5 * NN.dot(mixture_sum.dot(((invL+np.outer(a,a)).astype(V.dtype).dot(V) * (iE.flat * V)).sum(0)))
        L += 0.5 * (logdet(invL) - np.sum(np.diag(invL) + a**2, 0) + R)

    # Construct transition probability matrix with linear chain of 'minDur'
    # states for each of 'maxSpeaker' speaker. The last state in each chain has
    # self-loop probability 'loopProb' and the transition probabilities to the
    # initial chain states given by vector '(1-loopProb) * sp'. From all other,
    #states, one must move to the next state in the chain with probability one.
    tr[minDur-1::minDur,0::minDur]=(1-loopProb)*sp
    tr[(np.arange(1,maxSpeakers+1)*minDur-1,)*2] += loopProb
    ip[::minDur]=sp
    # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
    # for each speaker.
    q, tll, lf, lb = forward_backward(lls.repeat(minDur,axis=1), tr, ip) #, np.arange(1,maxSpeakers+1)*minDur-1)

    # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
    # L now contains -KL{q(Y)||p(Y)}. Therefore, L+ttl is correct value for ELBO.
    L += tll
    Li.append([L])

    # ML estimate of speaker prior probabilities (analogue to eq. (38))
    sp = q[0,::minDur] + np.exp(logsumexp(lf[:-1,minDur-1::minDur],axis=1)[:,np.newaxis]
                       + lb[1:,::minDur] + lls[1:] + np.log((1-loopProb)*sp)-tll).sum(0)
    sp = sp / sp.sum()

    # per-frame speaker posteriors (analogue to eq. (30)), obtained by summing
    # HMM state posteriors corresponding to each speaker
    q = q.reshape(len(q),maxSpeakers,minDur).sum(axis=2)


    # if reference is provided, report DER, cross-entropy and plot the figures
    if ref is not None:
      Li[-1] += [DER(downsampler.T.dot(q), ref), DER(downsampler.T.dot(q), ref, xentropy=True)]

      if plot:
        import matplotlib.pyplot
        if ii == 0: matplotlib.pyplot.clf()
        matplotlib.pyplot.subplot(maxIters, 1, ii+1)
        matplotlib.pyplot.plot(downsampler.T.dot(q), lw=2)
        matplotlib.pyplot.imshow(np.atleast_2d(ref), interpolation='none', aspect='auto',
                                 cmap=matplotlib.pyplot.cm.Pastel1, extent=(0, len(ref), -0.05, 1.05))
        
      print(ii, Li[-2])


    if ii > 0 and L - Li[-2][0] < epsilon:
      if L - Li[-1][0] < 0: print('WARNING: Value of auxiliary function has decreased!')
      break

  if downsample is not None:
    #upsample resulting q to match number of frames in the input utterance
    q = downsampler.T.dot(q)

  return q, sp, Li


def precalculate_VtiEV(V, iE):
    tril_ind = np.tril_indices(V.shape[0])
    VtiEV = np.empty((V.shape[1],len(tril_ind[0])), V.dtype)
    for c in range(V.shape[1]):
        VtiEV[c,:] = np.dot(V[:,c,:]*iE[np.newaxis,c,:], V[:,c,:].T)[tril_ind]
    return VtiEV


# Initialize q (per-frame speaker posteriors) from a reference
# (vector of per-frame zero based integer speaker IDs)
def frame_labels2posterior_mx(labels, maxSpeakers):
    #initialize from reference
    #pmx = np.zeros((len(labels), labels.max()+1))
    pmx = np.zeros((len(labels), maxSpeakers))
    pmx[np.arange(len(labels)), labels] = 1
    return pmx

# Calculates Diarization Error Rate (DER) or per-frame cross-entropy between
# reference (vector of per-frame zero based integer speaker IDs) and q (per-frame
# speaker posteriors). If expected=False, q is converted into hard labels before
# calculating DER. If expected=TRUE, posteriors in q are used to calculated
# "expected" DER.
def DER(q, ref, expected=True, xentropy=False):
    from itertools import permutations

    if not expected:
        # replce probabiities in q by zeros and ones
        hard_labels = q.argmax(1)
        q = np.zeros_like(q)
        q[range(len(q)), hard_labels] = 1

    err_mx = np.empty((ref.max()+1, q.shape[1]))
    for s in range(err_mx.shape[0]):
        tmpq = q[ref == s,:]
        err_mx[s] = (-np.log(tmpq) if xentropy else tmpq).sum(0)

    if err_mx.shape[0] < err_mx.shape[1]:
        err_mx = err_mx.T

    # try all alignments (permutations) of reference and detected speaker
    #could be written in more efficient way using dynamic programing
    acc = [err_mx[perm[:err_mx.shape[1]], range(err_mx.shape[1])].sum()
              for perm in permutations(range(err_mx.shape[0]))]
    if xentropy:
       return min(acc)/float(len(ref))
    else:
       return (len(ref) - max(acc))/float(len(ref))


###############################################################################
# Module private functions
###############################################################################
def logsumexp(x, axis=0):
    xmax = x.max(axis)
    x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    infs = np.isinf(xmax)
    if np.ndim(x) > 0:
      x[infs] = xmax[infs]
    elif infs:
      x = xmax
    return x


# The folowing two functions are only versions optimized for speed using numexpr
# module and can be replaced by logsumexp and np.exp functions to avoid
# the dependency on the module.
def logsumexp_ne(x, axis=0):
    xmax = np.array(x).max(axis=axis)
    xmax_e = np.expand_dims(xmax, axis)
    x = ne.evaluate("sum(exp(x - xmax_e), axis=%d)" % axis)
    x = ne.evaluate("xmax + log(x)")
    infs = np.isinf(xmax)
    if np.ndim(x) > 0:
      x[infs] = xmax[infs]
    elif infs:
      x = xmax
    return x


def exp_ne(x, out=None):
    return ne.evaluate("exp(x)", out=None)


# Convert vector with lower-triangular coefficients into symetric matrix
def tril_to_sym(tril):
    R = np.sqrt(len(tril)*2).astype(int)
    tril_ind = np.tril_indices(R)
    S = np.empty((R,R))
    S[tril_ind]       = tril
    S[tril_ind[::-1]] = tril
    return S


def logdet(A):
    return 2*np.sum(np.log(np.diag(spl.cholesky(A))))


def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. statrting in the state)
    Outputs:
        sp  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    ltr = np.log(tr)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip)
    lbw[-1] = 0.0

    for ii in range(1,len(lls)):
        lfw[ii] =  lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)

    tll = logsumexp(lfw[-1])
    sp = np.exp(lfw + lbw - tll)
    return sp, tll, lfw, lbw
