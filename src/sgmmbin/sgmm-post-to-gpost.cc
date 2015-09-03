// sgmmbin/sgmm-post-to-gpost.cc

// Copyright 2009-2012   Saarland University  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                2014   Guoguo Chen

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
#include "sgmm/estimate-am-sgmm.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Convert posteriors to Gaussian-level posteriors for SGMM training.\n"
        "Usage: sgmm-post-to-gpost [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <gpost-wspecifier>\n"
        "e.g.: sgmm-post-to-gpost 1.mdl 1.ali scp:train.scp 'ark:ali-to-post ark:1.ali ark:-|' ark:-";

    ParseOptions po(usage);
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    SgmmGselectConfig sgmm_opts;
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    sgmm_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        gpost_wspecifier = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);

    SgmmPerFrameDerivedVars per_frame_vars;

    SgmmGauPostWriter gpost_writer(gpost_wspecifier);

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        Posterior posterior = posteriors_reader.Value(utt);

        bool have_gselect  = !gselect_rspecifier.empty()
            && gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() == mat.NumRows();
        if (!gselect_rspecifier.empty() && !have_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
        std::vector<std::vector<int32> > empty_gselect;
        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (posterior.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }
        
        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_other_error++;
            continue;
          }
        }  // else spk_vars is "empty"

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        SgmmGauPost gpost(posterior.size());  // posterior.size() == T.

        SortPosteriorByPdfs(trans_model, &posterior);
        int32 prev_pdf_id = -1;
        BaseFloat prev_like = 0;
        Matrix<BaseFloat> prev_posterior;
        for (size_t i = 0; i < posterior.size(); i++) {

          std::vector<int32> this_gselect;
          if (!gselect->empty()) this_gselect = (*gselect)[i];
          else am_sgmm.GaussianSelection(sgmm_opts, mat.Row(i), &this_gselect);
          am_sgmm.ComputePerFrameVars(mat.Row(i), this_gselect, spk_vars, 0.0, &per_frame_vars);

          gpost[i].gselect = this_gselect;
          gpost[i].tids.resize(posterior[i].size());
          gpost[i].posteriors.resize(posterior[i].size());

          prev_pdf_id = -1;       // Only cache for the same frame.
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            gpost[i].tids[j] = tid;

            if (pdf_id != prev_pdf_id) {
              // First time see this pdf-id for this frame, update the cached
              // variables.
              prev_pdf_id = pdf_id;
              prev_like = am_sgmm.ComponentPosteriors(per_frame_vars, pdf_id,
                                                      &prev_posterior);
            }

            gpost[i].posteriors[j] = prev_posterior;
            tot_like_this_file += prev_like * weight;
            tot_weight += weight;
            gpost[i].posteriors[j].Scale(weight);
          }
        }

        KALDI_LOG << "Average like for this file is "
                  << (tot_like_this_file/posterior.size()) << " over "
                  << posterior.size() <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += posterior.size();
        if (num_done % 10 == 0)
          KALDI_LOG << "Avg like per frame so far is "
                    << (tot_like/tot_t);
        gpost_writer.Write(utt, gpost);
      }
    }

    KALDI_LOG << "Overall like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


