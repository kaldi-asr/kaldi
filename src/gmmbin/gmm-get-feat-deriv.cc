// gmmbin/gmm-get-feat-deriv.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/fmpe.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using kaldi::int32;
  try {
    const char *usage =
        "From GMM model and posteriors (which don't have to be positive),\n"
        "output for each utterance a matrix of likelihood derivatives w.r.t.\n"
        "the features.\n"
        "E.g. used in feature-space discriminative training.\n"
        "\n"
        "Usage:  gmm-get-feat-deriv [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <feature-deriv-wspecifier>\n"
        "e.g.: \n"
        " gmm-get-feat-deriv 1.mdl \"$feats\" ark:1.post ark:1.deriv\n";
        
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        deriv_wspecifier = po.GetArg(4);
    
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    BaseFloatMatrixWriter deriv_writer(deriv_wspecifier);
    
    int32 num_done = 0, num_err = 0;
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        KALDI_WARN << "No posteriors for utterance " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size " <<
              (posterior.size()) << " vs. "<< (mat.NumRows());
          num_err++;
          continue;
        }

        num_done++;

        // Derivative of likelihood (or whatever objective func.)
        // w.r.t. features.
        Matrix<BaseFloat> deriv;
        ComputeAmGmmFeatureDeriv(am_gmm, trans_model, posterior,
                                 mat, &deriv);
        
        deriv_writer.Write(key, deriv);
        if (num_done % 100 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances.";
      }        
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


