// chainbin/chain-make-num-fst-e2e.cc

// Copyright 2020  Yiwen Shao

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

/** @brief Converts fsts (containing transition-ids) to fsts (containing pdf-ids + 1).
*/
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

namespace kaldi {

bool FstTransitionToPdfPlusOne(const fst::StdVectorFst &fst_transition,
			       const TransitionModel &trans_model,
			       fst::StdVectorFst *fst_pdf) {
  fst::StdVectorFst fst_tmp(fst_transition);
  fst::RemoveEpsLocal(&fst_tmp);
  fst::RmEpsilon(&fst_tmp);
  // first change labels to pdf-id + 1
  int32 num_states = fst_tmp.NumStates();
  for (int32 state = 0; state < num_states; state++) {
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(&fst_tmp, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      if (arc.ilabel == 0) {
        KALDI_WARN << "Utterance rejected due to eps on input label";
        return false;
      }
      KALDI_ASSERT(arc.ilabel != 0);
      fst::StdArc arc2(arc);
      arc2.ilabel = arc2.olabel = trans_model.TransitionIdToPdf(arc.ilabel) + 1;
      aiter.SetValue(arc2);
    }
  }
  *fst_pdf = fst_tmp;
  return true;
}

bool AddWeightToFst(const fst::StdVectorFst &normalization_fst,
		    fst::StdVectorFst *fst) {
  // Note: by default, 'Compose' will call 'Connect', so if the
  // resulting FST is not connected, it will end up empty.
  fst::StdVectorFst composed_fst;
  fst::Compose(*fst, normalization_fst,
	       &composed_fst);
  *fst = composed_fst;
  if (composed_fst.NumStates() == 0)
    return false;
  return true;
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Converts chain e2e numerator fst (containing transition-ids) to fst (containing pdf-ids+1, \n"
      "and composed by the normalization fst) \n"
      "Usage:  chain-make-num-fst-e2e [options] <model> <normalization-fst>\n"
      "<trainsition-fst-rspecifier> <pdf-fst-wspecifier>\n"
        "e.g.: \n"
        " chain-make-num-fst-e2e 1.mdl ark:1.fst ark,t:-\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
      normalization_fst_rxfilename = po.GetArg(2),
      fsts_rspecifier = po.GetArg(3),
      fsts_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    fst::StdVectorFst normalization_fst;
    ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);

    SequentialTableReader<fst::VectorFstHolder> fsts_reader(fsts_rspecifier);
    TableWriter<fst::VectorFstHolder> fsts_writer(fsts_wspecifier);

    int32 num_done = 0;
    for (; !fsts_reader.Done(); fsts_reader.Next()) {
      std::string key = fsts_reader.Key();
      fst::VectorFst<fst::StdArc> fst_transition(fsts_reader.Value());
      fst::StdVectorFst fst_pdf;
      FstTransitionToPdfPlusOne(fst_transition, trans_model, &fst_pdf);
      AddWeightToFst(normalization_fst, &fst_pdf);
      fsts_writer.Write(key, fst_pdf);
      num_done++;
    }
    KALDI_LOG << "Converted " << num_done << " Fsts with transition-id to Fsts with pdf-id and normalized.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
