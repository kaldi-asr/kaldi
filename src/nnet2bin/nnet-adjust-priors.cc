// nnet2bin/nnet-adjust-priors.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

namespace kaldi {
namespace nnet2 {


// Computes one-sided K-L divergence from p to q.
BaseFloat KlDivergence(const Vector<BaseFloat> &p,
                       const Vector<BaseFloat> &q) {
  BaseFloat sum_p = p.Sum(), sum_q = q.Sum();
  if (fabs(sum_p - 1.0) > 0.01 || fabs(sum_q - 1.0) > 0.01) {
    KALDI_WARN << "KlDivergence: vectors are not close to being normalized "
               << sum_p << ", " << sum_q;
  }
  KALDI_ASSERT(p.Dim() == q.Dim());
  double ans = 0.0;

  for (int32 i = 0; i < p.Dim(); i++) {
    BaseFloat p_prob = p(i) / sum_p, q_prob = q(i) / sum_q;
    ans += p_prob * Log(p_prob / q_prob);
  }
  return ans;
}

void PrintPriorDiagnostics(const Vector<BaseFloat> &old_priors,
                           const Vector<BaseFloat> &new_priors) {
  if (old_priors.Dim() == 0) {
    KALDI_LOG << "Model did not previously have priors attached.";
  } else {
    Vector<BaseFloat> diff_prior(new_priors);
    diff_prior.AddVec(-1.0, old_priors);
    diff_prior.ApplyAbs();
    int32 max_index;
    diff_prior.Max(&max_index);
    KALDI_LOG << "Adjusting priors: largest absolute difference was for "
              << "pdf " << max_index << ", " << old_priors(max_index)
              << " -> " << new_priors(max_index);
    KALDI_LOG << "Adjusting priors: K-L divergence from old to new is "
              << KlDivergence(old_priors, new_priors);
  }
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Set the priors of the neural net to the computed posterios from the net,\n"
        "on typical data (e.g. training data). This is correct under more general\n"
        "circumstances than using the priors of the class labels in the training data\n"
        "\n"
        "Typical usage of this program will involve computation of an average pdf-level\n"
        "posterior with nnet-compute or nnet-compute-from-egs, piped into matrix-sum-rows\n"
        "and then vector-sum, to compute the average posterior\n"
        "\n"
        "Usage: nnet-adjust-priors [options] <nnet-in> <summed-posterior-vector-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-adjust-priors final.mdl prior.vec final.mdl\n";
    
    bool binary_write = true;
    BaseFloat prior_floor = 1.0e-15; // Have a very low prior floor, since this method
                                     // isn't likely to have a problem with very improbable
                                     // classes.
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("prior-floor", &prior_floor, "When setting priors, floor for "
                "priors (only used to avoid generating NaNs upon inversion)");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        posterior_vec_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }
    

    Vector<BaseFloat> posterior_vec;
    ReadKaldiObject(posterior_vec_rxfilename, &posterior_vec);

    KALDI_ASSERT(posterior_vec.Sum() > 0.0);
    posterior_vec.Scale(1.0 / posterior_vec.Sum()); // Renormalize
    
    Vector<BaseFloat> old_priors(am_nnet.Priors());

    PrintPriorDiagnostics(old_priors, posterior_vec);
    
    am_nnet.SetPriors(posterior_vec);
        
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Modified priors of neural network model and wrote it to "
              << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
