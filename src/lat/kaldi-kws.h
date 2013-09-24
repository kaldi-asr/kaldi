// lat/kaldi-kws.h

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#ifndef KALDI_LAT_KALDI_KWS_H_
#define KALDI_LAT_KALDI_KWS_H_

#include "fst/fstlib.h"
#include "lat/arctic-weight.h"

namespace kaldi {

using fst::TropicalWeight;
using fst::LogWeight;
using fst::ArcticWeight;

// The T*T*T semiring
typedef fst::LexicographicWeight<TropicalWeight, TropicalWeight> StdLStdWeight;
typedef fst::LexicographicWeight<TropicalWeight, StdLStdWeight> StdLStdLStdWeight;
typedef fst::ArcTpl<StdLStdLStdWeight> StdLStdLStdArc;

// The LxTxT' semiring
typedef fst::ProductWeight<TropicalWeight, ArcticWeight> StdXStdprimeWeight;
typedef fst::ProductWeight<LogWeight, StdXStdprimeWeight> LogXStdXStdprimeWeight;
typedef fst::ArcTpl<LogXStdXStdprimeWeight> LogXStdXStdprimeArc;

// Rename the weight and arc types to make them look more "friendly".
typedef StdLStdLStdWeight KwsLexicographicWeight;
typedef StdLStdLStdArc KwsLexicographicArc;
typedef fst::VectorFst<KwsLexicographicArc> KwsLexicographicFst;
typedef LogXStdXStdprimeWeight KwsProductWeight;
typedef LogXStdXStdprimeArc KwsProductArc;
typedef fst::VectorFst<KwsProductArc> KwsProductFst;
           

} // namespace kaldi

#endif  // KALDI_LAT_KALDI_KWS_H_
