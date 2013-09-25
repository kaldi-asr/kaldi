// gmm/model-common.h

// Copyright 2009-2012  Saarland University;  Microsoft Corporation;
//                      Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_GMM_MODEL_COMMON_H_
#define KALDI_GMM_MODEL_COMMON_H_
#include "matrix/matrix-lib.h"

namespace kaldi {

enum GmmUpdateFlags {
  kGmmMeans       = 0x001,  // m
  kGmmVariances   = 0x002,  // v
  kGmmWeights     = 0x004,  // w
  kGmmTransitions = 0x008,  // t ... not really part of GMM.
  kGmmAll       = 0x00F  // a
};
typedef uint16 GmmFlagsType;  ///< Bitwise OR of the above flags.
/// Convert string which is some subset of "mSwa" to
/// flags.
GmmFlagsType StringToGmmFlags(std::string str);

/// Convert GMM flags to string
std::string GmmFlagsToString(GmmFlagsType gmm_flags);

// Make sure that the flags make sense, i.e. if there is variance
// accumulation that there is also mean accumulation
GmmFlagsType AugmentGmmFlags(GmmFlagsType flags);

enum SgmmUpdateFlags {  /// The letters correspond to the variable names.
  kSgmmPhoneVectors                = 0x001,  /// v
  kSgmmPhoneProjections            = 0x002,  /// M
  kSgmmPhoneWeightProjections      = 0x004,  /// w
  kSgmmCovarianceMatrix            = 0x008,  /// S
  kSgmmSubstateWeights             = 0x010,  /// c
  kSgmmSpeakerProjections          = 0x020,  /// N
  kSgmmTransitions                 = 0x040,  /// t .. not really part of SGMM.
  kSgmmSpeakerWeightProjections    = 0x080,  /// u [ for SSGMM ]
  kSgmmAll                         = 0x0FF   /// a (won't normally use this).  
};

typedef uint16 SgmmUpdateFlagsType;  ///< Bitwise OR of the above flags.
SgmmUpdateFlagsType StringToSgmmUpdateFlags(std::string str);

enum SgmmWriteFlags {
  kSgmmGlobalParams    = 0x001,  /// g
  kSgmmStateParams     = 0x002,  /// s
  kSgmmNormalizers     = 0x004,  /// n
  kSgmmBackgroundGmms  = 0x008,  /// u
  kSgmmWriteAll        = 0x00F  /// a
};

typedef uint16 SgmmWriteFlagsType;  ///< Bitwise OR of the above flags.

SgmmWriteFlagsType StringToSgmmWriteFlags(std::string str);

/// Get Gaussian-mixture or substate-mixture splitting targets,
/// according to a power rule (e.g. typically power = 0.2).
/// Returns targets for number of mixture components (Gaussians,
/// or sub-states), allocating the Gaussians or whatever according
/// to a power of occupancy in order to acheive the total supplied
/// "target".  During splitting we ensure that
/// each Gaussian [or sub-state] would get a count of at least
/// "min-count", assuming counts were evenly distributed between
/// Gaussians in a state.
/// The vector "targets" will be resized to the appropriate dimension;
/// its value at input is ignored.
void GetSplitTargets(const Vector<BaseFloat> &state_occs,
                     int32 target_components,
                     BaseFloat power,
                     BaseFloat min_count,
                     std::vector<int32> *targets);

}  // End namespace kaldi


#endif  // KALDI_GMM_MODEL_COMMON_H_
