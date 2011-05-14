// gmm/model-common.h

// Copyright 2009-2011  Arnab Ghoshal  Microsoft Corporation

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

namespace kaldi {

enum GmmUpdateFlags {
  kGmmMeans     = 0x001,  // m
  kGmmVariances = 0x002,  // v
  kGmmWeights   = 0x004,  // w
  kGmmAll       = 0x007  // a
};
typedef uint16 GmmFlagsType;  ///< Bitwise OR of the above flags.
/// Convert string which is some subset of "mSwa" to
/// flags.
GmmFlagsType StringToGmmFlags(std::string str);

enum SgmmUpdateFlags {  /// The letters correspond to the variable names.
  kSgmmPhoneVectors       = 0x001,  /// v
  kSgmmPhoneProjections   = 0x002,  /// M
  kSgmmWeightProjections  = 0x004,  /// w
  kSgmmCovarianceMatrix   = 0x008,  /// S
  kSgmmSubstateWeights    = 0x010,  /// c
  kSgmmSpeakerProjections = 0x020,  /// N
  kSgmmAll                = 0x03F  /// a (won't normally use this).
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

SgmmUpdateFlagsType StringToSgmmWriteFlags(std::string str);

}  // End namespace kaldi


#endif  // KALDI_GMM_MODEL_COMMON_H_
