// kws/kws-functions.cc

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


#include "lat/lattice-functions.h"
#include "kws/kws-functions.h"
#include "fstext/determinize-star.h"
#include "fstext/epsilon-property.h"

// this file implements things in kws-functions.h; it's an overflow from
// kws-functions.cc (we split it up for compilation speed and to avoid
// generating too-large object files on cygwin).

namespace kaldi {


// This function replaces a symbol with epsilon wherever it appears
// (fst must be an acceptor).
template<class Arc>
static void ReplaceSymbolWithEpsilon(typename Arc::Label symbol,
                                     fst::VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  for (StateId s = 0; s < fst->NumStates(); s++) {
    for (fst::MutableArcIterator<fst::VectorFst<Arc> > aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      if (arc.ilabel == symbol) {
        arc.ilabel = 0;
        arc.olabel = 0;
        aiter.SetValue(arc);
      }
    }
  }
}

void DoFactorMerging(KwsProductFst *factor_transducer,
                     KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsProductFst::Arc::Label Label;

  // Encode the transducer first
  EncodeMapper<KwsProductArc> encoder(kEncodeLabels, ENCODE);
  Encode(factor_transducer, &encoder);


  // We want DeterminizeStar to remove epsilon arcs, so turn whatever it encoded
  // epsilons as, into actual epsilons.
  {
    KwsProductArc epsilon_arc(0, 0, KwsProductWeight::One(), 0);
    Label epsilon_label = encoder(epsilon_arc).ilabel;
    ReplaceSymbolWithEpsilon(epsilon_label, factor_transducer);
  }


  MaybeDoSanityCheck(*factor_transducer);

  // Use DeterminizeStar
  KALDI_VLOG(2) << "DoFactorMerging: determinization...";
  KwsProductFst dest_transducer;
  DeterminizeStar(*factor_transducer, &dest_transducer);

  MaybeDoSanityCheck(dest_transducer);

  // Commenting the minimization out, as it moves states/arcs in a way we don't
  // want in some rare cases. For example, if we have two arcs from starting
  // state, which have same words on the input side, but different cluster IDs
  // on the output side, it may make the two arcs sharing a common final arc,
  // which will cause problem in the factor disambiguation stage (we will not
  // be able to add disambiguation symbols for both paths). We do a final step
  // optimization anyway so commenting this out shouldn't matter too much.
  // KALDI_VLOG(2) << "DoFactorMerging: minimization...";
  // Minimize(&dest_transducer);

  MaybeDoSanityCheck(dest_transducer);

  Decode(&dest_transducer, encoder);

  Map(dest_transducer, index_transducer, KwsProductFstToKwsLexicographicFstMapper());
}

void DoFactorDisambiguation(KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsLexicographicArc::StateId StateId;

  StateId ns = index_transducer->NumStates();
  for (StateId s = 0; s < ns; s++) {
    for (MutableArcIterator<KwsLexicographicFst>
         aiter(index_transducer, s); !aiter.Done(); aiter.Next()) {
      KwsLexicographicArc arc = aiter.Value();
      if (index_transducer->Final(arc.nextstate) != KwsLexicographicWeight::Zero())
        arc.ilabel = s;
      else
        arc.olabel = 0;
      aiter.SetValue(arc);
    }
  }
}

void OptimizeFactorTransducer(KwsLexicographicFst *index_transducer,
                              int32 max_states,
                              bool allow_partial) {
  using namespace fst;
  KwsLexicographicFst ifst = *index_transducer;
  EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
  Encode(&ifst, &encoder);
  KALDI_VLOG(2) << "OptimizeFactorTransducer: determinization...";
  if (allow_partial) {
    DeterminizeStar(ifst, index_transducer, kDelta, NULL, max_states, true);
  } else {
      try {
        DeterminizeStar(ifst, index_transducer, kDelta, NULL, max_states,
                        false);
      } catch(const std::exception &e) {
        KALDI_WARN << e.what();
        *index_transducer = ifst;
      }
  }
  KALDI_VLOG(2) << "OptimizeFactorTransducer: minimization...";
  Minimize(index_transducer, static_cast<KwsLexicographicFst *>(NULL), fst::kDelta, true);
  Decode(index_transducer, encoder);
}

} // end namespace kaldi
