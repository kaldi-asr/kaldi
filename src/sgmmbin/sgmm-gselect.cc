// sgmmbin/sgmm-gselect.cc

// Copyright 2009-2011   Saarland University;  Microsoft Corporation

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Precompute Gaussian indices for SGMM training "
        "Usage: sgmm-gselect [options] <model-in> <feature-rspecifier> <gselect-wspecifier>\n"
        "e.g.: sgmm-gselect 1.sgmm \"ark:feature-command |\" ark:1.gs\n"
        "Note: you can do the same thing by combining the programs sgmm-write-ubm, fgmm-global-to-gmm,\n"
        "gmm-gselect and fgmm-gselect\n";

    ParseOptions po(usage);
    kaldi::SgmmGselectConfig sgmm_opts;
    std::string preselect_rspecifier;
    std::string likelihood_wspecifier;
    po.Register("preselect", &preselect_rspecifier, "Rspecifier for sets of Gaussians to "
                "limit gselect to (e.g. for gender dependent systems)");
    po.Register("write-likes", &likelihood_wspecifier, "Wspecifier for likelihoods per "
                "utterance");
    sgmm_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gselect_wspecifier = po.GetArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_filename, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorVectorWriter gselect_writer(gselect_wspecifier);
    BaseFloatWriter likelihood_writer(likelihood_wspecifier);
    RandomAccessInt32VectorReader preselect_reader(preselect_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      int32 tot_t_this_file = 0; double tot_like_this_file = 0;
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      std::vector<std::vector<int32> > gselect_vec(mat.NumRows());
      tot_t_this_file += mat.NumRows();
      if(preselect_rspecifier != "") { // e.g. gender dependent.        
        if (!preselect_reader.HasKey(utt)) {
          KALDI_WARN << "No preselect information for utterance " << utt;
          num_err++;
          continue;
        }
        const std::vector<int32> &preselect = preselect_reader.Value(utt);
        KALDI_ASSERT(!preselect.empty());
        for (int32 i = 0; i < mat.NumRows(); i++)
          tot_like_this_file +=
              am_sgmm.GaussianSelectionPreselect(sgmm_opts, mat.Row(i),
                                                 preselect, &(gselect_vec[i]));
      } else {
        for (int32 i = 0; i < mat.NumRows(); i++)
          tot_like_this_file += am_sgmm.GaussianSelection(sgmm_opts, mat.Row(i), &(gselect_vec[i]));
      }
      gselect_writer.Write(utt, gselect_vec);
      if (num_done % 10 == 0)
        KALDI_LOG << "For " << num_done << "'th file, average UBM likelihood over "
                  << tot_t_this_file << " frames is "
                  << (tot_like_this_file/tot_t_this_file);
      tot_t += tot_t_this_file;
      tot_like += tot_like_this_file;

      if(likelihood_wspecifier != "")
        likelihood_writer.Write(utt, tot_like_this_file);
      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors, average UBM log-likelihood is "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";


    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


