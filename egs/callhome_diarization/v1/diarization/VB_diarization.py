#!/usr/bin/env python

# Copyright 2013-2019 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
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
#   16/07/13 01:00AM - original version
#   20/06/17 12:07AM - np.asarray replaced by .toarray()
#                    - minor bug fix in initializing q(Z)
#                    - minor bug fix in ELBO calculation
#                    - few more optimizations
#   03/10/19 02:27PM - speaker regularization coefficient Fb added
#

import numpy as np
from scipy.sparse import coo_matrix
import scipy.linalg as spl
import numexpr as ne # the dependency on this modul can be avoided by replacing
                     # logsumexp_ne and exp_ne with logsumexp and np.exp

#[gamma pi Li] =
def VB_diarization(X, m, invSigma, w, V, pi=None, gamma=None,
                   maxSpeakers = 10, maxIters = 10,
                   epsilon = 1e-4, loopProb = 0.99, statScale = 1.0,
                   alphaQInit = 1.0, downsample = None, VtinvSigmaV = None, ref=None,
                   plot=False, sparsityThr=0.001, llScale=1.0, minDur=1, Fa=1.0, Fb=1.0):

  """
  This a generalized version of speaker diarization described in:

  Diez. M., Burget. L., Landini. F., Cernocky. J.
  Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors

  Variable names and equation numbers refer to those used the paper

  Inputs:
  X  - T x D array, where columns are D dimensional feature vectors for T frames
  m  - C x D array of GMM component means
  invSigma - C x D array of GMM component inverse covariance matrix diagonals
  w  - C dimensional column vector of GMM component weights
  V  - R x C x D array of eigenvoices
  maxSpeakers - maximum number of speakers expected in the utterance
  maxIters    - maximum number of algorithm iterations
  epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
  loopProb    - probability of not switching speakers between frames
  statScale   - deprecated, use Fa instead
  Fa          - scale sufficient statiscits collected using UBM
  Fb          - speaker regularization coefficient Fb (controls final # of speaker)
  llScale     - scale UBM likelihood (i.e. llScale < 1.0 make atribution of
                frames to UBM componets more uncertain)
  sparsityThr - set occupations smaller that this threshold to 0.0 (saves memory
                as the posteriors are represented by sparse matrix)
  alphaQInit  - Dirichlet concentraion parameter for initializing gamma
  downsample  - perform diarization on input downsampled by this factor
  VtinvSigmaV       - C x (R**2+R)/2 matrix normally calculated by VB_diarization when
                VtinvSigmaV is None. However, it can be pre-calculated using function
                precalculate_VtinvSigmaV(V) and used across calls of VB_diarization.
  minDur      - minimum number of frames between speaker turns imposed by linear
                chains of HMM states corresponding to each speaker. All the states
                in a chain share the same output distribution
  ref         - T dim. integer vector with reference speaker ID (0:maxSpeakers)
                per frame
  plot        - if set to True, plot per-frame speaker posteriors.

   Outputs:
   gamma  - S x T matrix of posteriors attribution each frame to one of S possible
        speakers, where S is given by opts.maxSpeakers
   pi - S dimensional column vector of ML learned speaker priors. Ideally, these
        should allow to estimate # of speaker in the utterance as the
        probabilities of the redundant speaker should converge to zero.
   Li - values of auxiliary function (and DER and frame cross-entropy between gamma  
        and reference if 'ref' is provided) over iterations.
  """

  # The references to equations corresponds to
  # Diez. M., Burget. L., Landini. F., Cernocky. J.
  # Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors

  D=X.shape[1]  # feature dimensionality
  C=len(w)      # number of mixture components
  R=V.shape[0]  # subspace rank
  nframes=X.shape[0]

  if VtinvSigmaV is None:
    VtinvSigmaV = precalculate_VtinvSigmaV(V, invSigma)

  V = V.reshape(V.shape[0],-1)

  if pi is None:
    pi = np.ones(maxSpeakers)/maxSpeakers
  else:
    maxSpeakers = len(pi)

  if gamma is None:
    # initialize gamma from flat Dirichlet prior with concentration parameter alphaQInit
    gamma = np.random.gamma(alphaQInit, size=(nframes, maxSpeakers))
    gamma = gamma / gamma.sum(1, keepdims=True)

  # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics)
  ll = (X**2).dot(-0.5*invSigma.T) + X.dot(invSigma.T*m.T)-0.5*((invSigma * m**2 - np.log(invSigma)).sum(1) - 2*np.log(w) + D*np.log(2*np.pi))
  ll *= llScale
  G = logsumexp_ne(ll, axis=1) 
  zeta =  exp_ne(ll - G[:,np.newaxis])  
  zeta[zeta<sparsityThr] = 0.0
  zeta = zeta * statScale
  G = G * statScale

  #Kx = np.sum(zeta * (np.log(w) - np.log(zeta)), 1)
  zeta = coo_matrix(zeta) # represent zero-order stats using sparse matrix
  print('Sparsity: ', len(zeta.row), float(len(zeta.row))/np.prod(zeta.shape))
  LL = np.sum(G) # total log-likelihod as calculated using UBM

  mixture_sum = coo_matrix((np.ones(C*D), (np.repeat(range(C),D), range(C*D))))

  #G = np.sum((zeta.multiply(ll - np.log(w))).toarray(), 1) + Kx  # from eq. (30) # Aleready calculated above

  # Calculate per-frame first order statistics projected into the R-dim. subspace
  # V^T \Sigma^{-1} F_m
  F_s =coo_matrix((((X[zeta.row]-m[zeta.col])*zeta.data[:,np.newaxis]).flat,
                   (zeta.row.repeat(D), zeta.col.repeat(D)*D+np.tile(range(D), len(zeta.col)))), shape=(nframes, D*C))
  rho = F_s.tocsr().dot((invSigma.flat * V).T)
  del F_s
  ## The code above is only efficient implementation of the following comented code
  #rho = 0;
  #for ii in range(C):
  #  rho = rho + V[ii*D:(ii+1)*D,:].T.dot(zeta[ii,:] * invSigma[:,[ii]] *  (X - m[:,[ii]]))

  if downsample is not None:
    # Downsample zeta, rho, G and gamma by summing the statistic over 'downsample' frames
    # This speeds-up diarization for the price of lowering its frame resolution
    #downsampler = coo_matrix((np.ones(nframes), (np.ceil(np.arange(nframes)/downsample).astype(int), np.arange(nframes))))
    downsampler = coo_matrix((np.ones(nframes), (np.ceil(np.arange(nframes)/downsample).astype(int), np.arange(nframes))))
    zeta  = downsampler.dot(zeta)
    rho   = downsampler.dot(rho)
    G     = downsampler.dot(G)
    gamma = downsampler.dot(gamma) / downsample
  else:
    downsampler=np.array(1)

  Li = [[LL*Fa]] # for the 0-th iteration,
  if ref is not None:
    Li[-1] += [DER(downsampler.T.dot(gamma), ref), DER(downsampler.T.dot(gamma), ref, xentropy=True)]

  ln_p = np.zeros_like(gamma)
  tr = np.eye(minDur*maxSpeakers, k=1)
  ip = np.zeros(minDur*maxSpeakers)
  for ii in range(maxIters):
    ELBO = 0                                                                   # objective function (11) (i.e. VB lower-bound on the evidence)
    sum_gamma_zeta =   zeta.T.dot(gamma).T                                     # corresponds to the last sum in eq. (26) for all 's'
    invLnoI_flat = sum_gamma_zeta.astype(VtinvSigmaV.dtype).dot(VtinvSigmaV)   # eq. (26) except for 'I' and the F_A F_B factors for all 's'
    sum_gamma_rho = gamma.T.dot(rho)                                           # summation in eq. (17) 
    for sid in range(maxSpeakers):
        invL = np.linalg.inv(np.eye(R) + tril_to_sym(invLnoI_flat[sid])*Fa/Fb) # eq. (18) inverse
        a = invL.dot(sum_gamma_rho[sid])*Fa/Fb                                 # eq. (17)
        ln_p[:,sid] = Fa * (G + rho.dot(a) - 0.5 * zeta.dot(mixture_sum.dot(((invL+np.outer(a,a)).astype(V.dtype).dot(V) * (invSigma.flat * V)).sum(0)))) #eq. (23)
        ELBO += Fb* 0.5 * (logdet(invL) - np.sum(np.diag(invL) + a**2, 0) + R)

    # Construct transition probability matrix with linear chain of 'minDur'
    # states for each of 'maxSpeaker' speaker. The last state in each chain has
    # self-loop probability 'loopProb' and the transition probabilities to the
    # initial chain states given by vector '(1-loopProb) * pi'. From all other,
    # states, one must move to the next state in the chain with probability one.
    tr[minDur-1::minDur,0::minDur]=(1-loopProb)*pi
    tr[(np.arange(1,maxSpeakers+1)*minDur-1,)*2] += loopProb
    ip[::minDur]=pi
    # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
    # for each speaker.
    gamma, tll, lf, lb = forward_backward(ln_p.repeat(minDur,axis=1), tr, ip) #, np.arange(1,maxSpeakers+1)*minDur-1)

    # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
    # ELBO now contains -KL{q(Y)||p(Y)}. Therefore, ELBO+ttl is correct value for ELBO.
    ELBO += tll
    Li.append([ELBO])

    # ML estimate of speaker prior probabilities, eq. (24)
    pi = gamma[0,::minDur] + np.exp(logsumexp(lf[:-1,minDur-1::minDur],axis=1)[:,np.newaxis]
                           + lb[1:,::minDur] + ln_p[1:] + np.log((1-loopProb)*pi)-tll).sum(0)
    pi = pi / pi.sum()

    # per-frame speaker posteriors (eq. (19)), obtained by summing
    # HMM state posteriors corresponding to each speaker
    gamma = gamma.reshape(len(gamma),maxSpeakers,minDur).sum(axis=2)


    # if reference is provided, report DER, cross-entropy and plot the figures
    if ref is not None:
      Li[-1] += [DER(downsampler.T.dot(gamma), ref), DER(downsampler.T.dot(gamma), ref, xentropy=True)]

      if plot:
        import matplotlib.pyplot
        if ii == 0: 
          matplotlib.pyplot.clf()
        matplotlib.pyplot.subplot(maxIters, 1, ii+1)
        matplotlib.pyplot.plot(downsampler.T.dot(gamma), lw=2)
        matplotlib.pyplot.imshow(np.atleast_2d(ref), interpolation='none', aspect='auto',
                                 cmap=matplotlib.pyplot.cm.Pastel1, extent=(0, len(ref), -0.05, 1.05))
      print(ii, Li[-2])


    if ii > 0 and ELBO - Li[-2][0] < epsilon:
      if ELBO - Li[-1][0] < 0: 
        print('WARNING: Value of auxiliary function has decreased!')
      break

  if downsample is not None:
    # upsample resulting gamma to match number of frames in the input utterance
    gamma = downsampler.T.dot(gamma)

  return gamma, pi, Li


def precalculate_VtinvSigmaV(V, invSigma):
    tril_ind = np.tril_indices(V.shape[0])
    VtinvSigmaV = np.empty((V.shape[1],len(tril_ind[0])), V.dtype)
    for c in range(V.shape[1]):
        VtinvSigmaV[c,:] = np.dot(V[:,c,:]*invSigma[np.newaxis,c,:], V[:,c,:].T)[tril_ind]
    return VtinvSigmaV


# Initialize q (per-frame speaker posteriors) from a reference
# (vector of per-frame zero based integer speaker IDs)
def frame_labels2posterior_mx(labels, maxSpeakers=None):
    #initialize from reference
    if maxSpeakers:
      pmx = np.zeros((len(labels), maxSpeakers))
    else:
      pmx = np.zeros((len(labels), labels.max()+1))
    pmx[np.arange(len(labels)), labels] = 1
    return pmx


# Calculates Diarization Error Rate (DER) or per-frame cross-entropy between
# reference (vector of per-frame zero based integer speaker IDs) and gamma 
# (per-frame speaker posteriors). If expected=False, gamma is converted into 
# hard labels before calculating DER. If expected=TRUE, posteriors in gamma 
# are used to calculate "expected" DER.
def DER(gamma, ref, expected=True, xentropy=False):
    from itertools import permutations

    if not expected:
        # replace probabiities in gamma by zeros and ones
        hard_labels = gamma.argmax(1)
        gamma = np.zeros_like(gamma)
        gamma[range(len(gamma)), hard_labels] = 1

    err_mx = np.empty((ref.max()+1, gamma.shape[1]))
    for s in range(err_mx.shape[0]):
        tmpgamma = gamma[ref == s,:]
        err_mx[s] = (-np.log(tmpgamma) if xentropy else tmpgamma).sum(0)

    if err_mx.shape[0] < err_mx.shape[1]:
        err_mx = err_mx.T

    # try all alignments (permutations) of reference and detected speaker
    # could be written in more efficient way using dynamic programing
    acc = [err_mx[perm[:err_mx.shape[1]], range(err_mx.shape[1])].sum()
              for perm in permutations(range(err_mx.shape[0]))]
    if xentropy:
       return min(acc)/float(len(ref))
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

    for ii in  range(1,len(lls)):
        lfw[ii] =  lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)

    tll = logsumexp(lfw[-1])
    sp = np.exp(lfw + lbw - tll)
    return sp, tll, lfw, lbw