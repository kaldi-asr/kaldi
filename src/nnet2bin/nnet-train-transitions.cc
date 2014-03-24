// nnet2bin/nnet-train-transitions.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

namespace kaldi {
namespace nnet2 {
void SetPriors(const TransitionModel &tmodel,
               const Vector<double> &transition_accs,
               double prior_floor,
               AmNnet *am_nnet) {
  KALDI_ASSERT(tmodel.NumPdfs() == am_nnet->NumPdfs());
  Vector<BaseFloat> pdf_counts(tmodel.NumPdfs());
  KALDI_ASSERT(transition_accs(0) == 0.0); // There is
  // no zero transition-id.
  for (int32 tid = 1; tid < transition_accs.Dim(); tid++) {
    int32 pdf = tmodel.TransitionIdToPdf(tid);
    pdf_counts(pdf) += transition_accs(tid);
  }
  BaseFloat sum = pdf_counts.Sum();
  KALDI_ASSERT(sum != 0.0);
  KALDI_ASSERT(prior_floor > 0.0 && prior_floor < 1.0);
  pdf_counts.Scale(1.0 / sum);
  pdf_counts.ApplyFloor(prior_floor);
  pdf_counts.Scale(1.0 / pdf_counts.Sum()); // normalize again.
  am_nnet->SetPriors(pdf_counts);
}               


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train the transition probabilities of a neural network acoustic model\n"
        "\n"
        "Usage:  nnet-train-transitions [options] <nnet-in> <alignments-rspecifier> <nnet-out>\n"
        "e.g.:\n"
        " nnet-train-transitions 1.nnet \"ark:gunzip -c ali.*.gz|\" 2.nnet\n";
    
    bool binary_write = true;
    bool set_priors = true; // Also set the per-pdf priors in the model.
    BaseFloat prior_floor = 5.0e-06; // The default was previously 1e-8, but
                                     // once we had problems with a pdf-id that
                                     // was not being seen in training, being
                                     // recognized all the time.  This value
                                     // seemed to be the smallest prior of the
                                     // "seen" pdf-ids in one run.
    MleTransitionUpdateConfig transition_update_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("set-priors", &set_priors, "If true, also set priors in neural "
                "net (we divide by these in test time)");
    po.Register("prior-floor", &prior_floor, "When setting priors, floor for "
                "priors");
    transition_update_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        ali_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }
    
    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);

    int32 num_done = 0;
    SequentialInt32VectorReader ali_reader(ali_rspecifier);
    for (; ! ali_reader.Done(); ali_reader.Next()) {
      const std::vector<int32> alignment(ali_reader.Value());
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];
        BaseFloat weight = 1.0;
        trans_model.Accumulate(weight, tid, &transition_accs);
      }
      num_done++;
    }
    KALDI_LOG << "Accumulated transition stats from " << num_done
              << " utterances.";

    {
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, transition_update_config,
                            &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << count
                << " frames.";
    }

    if (set_priors) {
      KALDI_LOG << "Setting priors of pdfs in the model.";
      SetPriors(trans_model, transition_accs, prior_floor, &am_nnet);
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Trained transitions of neural network model and wrote it to "
              << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
