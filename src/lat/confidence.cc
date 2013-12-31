// lat/confidence.cc

// Copyright 2013  Johns Hopkins University (Author: Daniel Povey)

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

#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {

BaseFloat SentenceLevelConfidence(const CompactLattice &clat,
                                  int32 *num_paths,
                                  std::vector<int32> *best_sentence,
                                  std::vector<int32> *second_best_sentence) {
  /* It may seem strange that the first thing we do is to convert the
     CompactLattice to a Lattice, given that we may have just created the
     CompactLattice by determinizing a Lattice.  However, this is not just
     a circular conversion; "lat" will have the property that distinct
     paths have distinct word sequences.
     Below, we could run NbestAsFsts on a CompactLattice, but the time
     taken would be quadratic in the length in words of the CompactLattice,
     because of the alignment information getting appended as vectors.
     That's why we convert back to Lattice.
  */
  Lattice lat;
  ConvertLattice(clat, &lat);
  std::vector<Lattice> lats;
  NbestAsFsts(lat, 2, &lats);
  int32 n = lats.size();
  KALDI_ASSERT(n >= 0 && n <= 2);
  if (num_paths != NULL) *num_paths = n;
  if (best_sentence != NULL) best_sentence->clear();
  if (second_best_sentence != NULL) second_best_sentence->clear();

  LatticeWeight weight1, weight2;
  if (n >= 1)
    fst::GetLinearSymbolSequence<LatticeArc,int32>(lats[0], NULL,
                                                   best_sentence,
                                                   &weight1);
  if (n >= 2)
    fst::GetLinearSymbolSequence<LatticeArc,int32>(lats[1], NULL,
                                                   second_best_sentence,
                                                   &weight2);

  if (n == 0) {
    return 0; // this seems most appropriate because it will be interpreted as
              // zero confidence, and something definitely went wrong for this
              // to happen.
  } else if (n == 1) {
    // If there is only one sentence in the lattice, we interpret this as there
    // being perfect confidence
    return std::numeric_limits<BaseFloat>::infinity();
  } else {
    BaseFloat best_cost = ConvertToCost(weight1),
        second_best_cost = ConvertToCost(weight2);
    BaseFloat ans = second_best_cost - best_cost;
    if (!(ans >= -0.001 * (fabs(best_cost) + fabs(second_best_cost)))) {
      // Answer should be positive.  Make sure it's at at least not
      // substantially negative.  This would be very strange.
      KALDI_WARN << "Very negative difference." << ans;
    }
    if (ans < 0) ans = 0;
    return ans;
  }  
}



BaseFloat SentenceLevelConfidence(const Lattice &lat,
                                  int32 *num_paths,
                                  std::vector<int32> *best_sentence,
                                  std::vector<int32> *second_best_sentence) {
  int32 max_sentence_length = LongestSentenceLength(lat);
  fst::DeterminizeLatticePrunedOptions determinize_opts;
  // safety_factor is just in case I forgot some reason why we might need a
  // couple extra arcs.  Setting it to 4 for extra safety costs very little.
  int32 safety_factor = 4;
  determinize_opts.max_arcs = max_sentence_length * 2 + safety_factor;
  // set prune_beam to a large value... we don't really rely on the beam; we
  // rely on the max_arcs variable to limit the size of the lattice.
  BaseFloat prune_beam = 1000.0; 

  CompactLattice clat;
  // We ignore the return status of DeterminizeLatticePruned.  It will likely
  // return false, but this is expected because the expansion is limited
  // by "max_arcs" not "prune_beam".
  DeterminizeLatticePruned(lat, prune_beam, &clat, determinize_opts);

  // Call the version of this function that takes a CompactLattice.
  return SentenceLevelConfidence(clat, num_paths,
                                 best_sentence, second_best_sentence);
}



}  // namespace kaldi
