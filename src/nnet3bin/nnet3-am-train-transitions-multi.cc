// nnet3bin/nnet3-am-train-transitions.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/am-nnet-multi.h"
#include "tree/context-dep.h"

namespace kaldi {
namespace nnet3 {
void SetPriors(const vector<TransitionModel*> &tmodel,
               const vector<Vector<double> > &transition_accs_vec,
               double prior_floor,
               AmNnetMulti *am_nnet) {
  KALDI_ASSERT(tmodel.size() == transition_accs_vec.size());
  int num_outputs = tmodel.size();
  for (int i = 0; i < num_outputs; i++) {
    KALDI_ASSERT(tmodel[i]->NumPdfs() == am_nnet->NumPdfs(i));

    Vector<BaseFloat> pdf_counts(tmodel[i]->NumPdfs());
    KALDI_ASSERT(transition_accs_vec[i](0) == 0.0); // There is
    // no zero transition-id.
    for (int32 tid = 1; tid < transition_accs_vec[i].Dim(); tid++) {
      int32 pdf = tmodel[i]->TransitionIdToPdf(tid);
      pdf_counts(pdf) += transition_accs_vec[i](tid);
    }
    BaseFloat sum = pdf_counts.Sum();
    KALDI_ASSERT(sum != 0.0);
    KALDI_ASSERT(prior_floor > 0.0 && prior_floor < 1.0);
    pdf_counts.Scale(1.0 / sum);
    pdf_counts.ApplyFloor(prior_floor);
    pdf_counts.Scale(1.0 / pdf_counts.Sum()); // normalize again.
    am_nnet->SetPriors(pdf_counts, i);
  }
}               


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train the transition probabilities of an nnet3 neural network acoustic model\n"
        "\n"
        "Usage:  nnet3-am-train-transitions [options] <nnet-in> <alignments-rspecifier> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-am-train-transitions 1.nnet \"ark:gunzip -c ali.*.gz|\" 2.nnet\n";
    
    bool binary_write = true;
    bool set_priors = true; // Also set the per-pdf priors in the model.
    BaseFloat prior_floor = 5.0e-06; // The default was previously 1e-8, but
                                     // once we had problems with a pdf-id that
                                     // was not being seen in training, being
                                     // recognized all the time.  This value
                                     // seemed to be the smallest prior of the
                                     // "seen" pdf-ids in one run.
    MleTransitionUpdateConfig transition_update_config;
    int num_outputs = 2;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("set-priors", &set_priors, "If true, also set priors in neural "
                "net (we divide by these in test time)");
    po.Register("prior-floor", &prior_floor, "When setting priors, floor for "
                "priors");
    transition_update_config.Register(&po);
//    po.Register("num-outputs", &num_outputs, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);
    std::string nnet_wxfilename = po.GetArg(po.NumArgs());
    
    vector<std::string> ali_rspecifiers;
    for (int i = 2; i < po.NumArgs(); i++) {
      ali_rspecifiers.push_back(po.GetArg(i));
    }

//    KALDI_ASSERT(ali_rspecifiers.size() == num_outputs);

    vector<TransitionModel*> trans_models;

    AmNnetMulti am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      ReadBasicType(ki.Stream(), binary_read, &num_outputs);
      for (int i = 0; i < num_outputs; i++) {
        TransitionModel* trans_model = new TransitionModel();
        trans_model->Read(ki.Stream(), binary_read);
        trans_models.push_back(trans_model);
      }
      am_nnet.Read(ki.Stream(), binary_read);
    }
    
    vector<Vector<double> > transition_accs_vec(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      trans_models[i]->InitStats(&transition_accs_vec[i]);
    }

    for (int n = 0; n < num_outputs; n++) {
      int32 num_done = 0;
      SequentialInt32VectorReader ali_reader(ali_rspecifiers[n]);
      for (; ! ali_reader.Done(); ali_reader.Next()) {
        const std::vector<int32> alignment(ali_reader.Value());
        for (size_t i = 0; i < alignment.size(); i++) {
          int32 tid = alignment[i];
          BaseFloat weight = 1.0;
          trans_models[n]->Accumulate(weight, tid, &transition_accs_vec[n]);
        }
        num_done++;
      }
      KALDI_LOG << "Accumulated transition stats from " << num_done
                << " utterances.";
    }

    for (int i = 0; i < num_outputs; i++) {
      BaseFloat objf_impr, count;
      trans_models[i]->MleUpdate(transition_accs_vec[i],
          transition_update_config,
                            &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << count
                << " frames.";
    }

    if (set_priors) {
      KALDI_LOG << "Setting priors of pdfs in the model.";
      SetPriors(trans_models, transition_accs_vec, prior_floor, &am_nnet);
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      WriteBasicType(ko.Stream(), binary_write, num_outputs);
      for (int i = 0; i < num_outputs; i++) {
        trans_models[i]->Write(ko.Stream(), binary_write);
      }
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
