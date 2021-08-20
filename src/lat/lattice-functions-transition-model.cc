// lat/lattice-functions-transition-model.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2013  Cisco Systems (author: Neha Agrawal) [code modified
//                      from original code in ../gmmbin/gmm-rescore-lattice.cc]
//                2014  Guoguo Chen

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

#include "lat/lattice-functions-transition-model.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

BaseFloat LatticeForwardBackwardMmi(
    const TransitionModel &tmodel,
    const Lattice &lat,
    const std::vector<int32> &num_ali,
    bool drop_frames,
    bool convert_to_pdf_ids,
    bool cancel,
    Posterior *post) {
  // First compute the MMI posteriors.

  Posterior den_post;
  BaseFloat ans = LatticeForwardBackward(lat,
                                         &den_post,
                                         NULL);

  Posterior num_post;
  AlignmentToPosterior(num_ali, &num_post);

  // Now negate the MMI posteriors and add the numerator
  // posteriors.
  ScalePosterior(-1.0, &den_post);

  if (convert_to_pdf_ids) {
    Posterior num_tmp;
    ConvertPosteriorToPdfs(tmodel, num_post, &num_tmp);
    num_tmp.swap(num_post);
    Posterior den_tmp;
    ConvertPosteriorToPdfs(tmodel, den_post, &den_tmp);
    den_tmp.swap(den_post);
  }

  MergePosteriors(num_post, den_post,
                  cancel, drop_frames, post);

  return ans;
}


bool CompactLatticeToWordProns(
    const TransitionModel &tmodel,
    const CompactLattice &clat,
    std::vector<int32> *words,
    std::vector<int32> *begin_times,
    std::vector<int32> *lengths,
    std::vector<std::vector<int32> > *prons,
    std::vector<std::vector<int32> > *phone_lengths) {
  words->clear();
  begin_times->clear();
  lengths->clear();
  prons->clear();
  phone_lengths->clear();
  typedef CompactLattice::Arc Arc;
  typedef Arc::Label Label;
  typedef CompactLattice::StateId StateId;
  typedef CompactLattice::Weight Weight;
  using namespace fst;
  StateId state = clat.Start();
  int32 cur_time = 0;
  if (state == kNoStateId) {
    KALDI_WARN << "Empty lattice.";
    return false;
  }
  while (1) {
    Weight final = clat.Final(state);
    size_t num_arcs = clat.NumArcs(state);
    if (final != Weight::Zero()) {
      if (num_arcs != 0) {
        KALDI_WARN << "Lattice is not linear.";
        return false;
      }
      if (! final.String().empty()) {
        KALDI_WARN << "Lattice has alignments on final-weight: probably "
            "was not word-aligned (alignments will be approximate)";
      }
      return true;
    } else {
      if (num_arcs != 1) {
        KALDI_WARN << "Lattice is not linear: num-arcs = " << num_arcs;
        return false;
      }
      fst::ArcIterator<CompactLattice> aiter(clat, state);
      const Arc &arc = aiter.Value();
      Label word_id = arc.ilabel; // Note: ilabel==olabel, since acceptor.
      // Also note: word_id may be zero; we output it anyway.
      int32 length = arc.weight.String().size();
      words->push_back(word_id);
      begin_times->push_back(cur_time);
      lengths->push_back(length);
      const std::vector<int32> &arc_alignment = arc.weight.String();
      std::vector<std::vector<int32> > split_alignment;
      SplitToPhones(tmodel, arc_alignment, &split_alignment);
      std::vector<int32> phones(split_alignment.size());
      std::vector<int32> plengths(split_alignment.size());
      for (size_t i = 0; i < split_alignment.size(); i++) {
        KALDI_ASSERT(!split_alignment[i].empty());
        phones[i] = tmodel.TransitionIdToPhone(split_alignment[i][0]);
        plengths[i] = split_alignment[i].size();
      }
      prons->push_back(phones);
      phone_lengths->push_back(plengths);

      cur_time += length;
      state = arc.nextstate;
    }
  }
}

} // end namespace kaldi
