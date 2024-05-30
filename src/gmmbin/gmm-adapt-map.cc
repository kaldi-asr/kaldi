// gmmbin/gmm-adapt-map.cc

// Copyright 2012  Cisco Systems (author: Neha Agrawal)
//                 Johns Hopkins University (author: Daniel Povey)
//           2014  Guoguo Chen

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

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Compute MAP estimates per-utterance (default) or per-speaker for\n"
        "the supplied set of speakers (spk2utt option).  This will typically\n"
        "be piped into gmm-latgen-map\n"
        "\n"
        "Usage: gmm-adapt-map  [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <map-am-wspecifier>\n";

    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    bool binary = true;
    MapDiagGmmOptions map_config;
    std::string update_flags_str = "mw";

    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters will be "
                "updated: subset of mvw.");
    map_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        map_am_wspecifier = po.GetArg(4);

    GmmFlagsType update_flags = StringToGmmFlags(update_flags_str);

    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    MapAmDiagGmmWriter map_am_writer(map_am_wspecifier);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }

    double tot_like = 0.0, tot_like_change = 0.0, tot_t = 0.0,
        tot_t_check = 0.0;
    int32 num_done = 0, num_err = 0;

    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        std::string spk = spk2utt_reader.Key();
        AmDiagGmm copy_am_gmm;
        copy_am_gmm.CopyFromAmDiagGmm(am_gmm);
        AccumAmDiagGmm map_accs;
        map_accs.Init(am_gmm, update_flags);

        const std::vector<std::string> &uttlist = spk2utt_reader.Value();

        // for each speaker, estimate MAP means
        std::vector<std::string>::const_iterator iter = uttlist.begin(),
            end = uttlist.end();
        for (; iter != end; ++iter) {
          std::string utt = *iter;
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          if (!posteriors_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &posterior = posteriors_reader.Value(utt);
          if (posterior.size() != feats.NumRows()) {
            KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
                       << " vs. " << (feats.NumRows());
            num_err++;
            continue;
          }

          BaseFloat file_like = 0.0, file_t = 0.0;
          Posterior pdf_posterior;
          ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
          for ( size_t i = 0; i < posterior.size(); i++ ) {
            for ( size_t j = 0; j < pdf_posterior[i].size(); j++ ) {
              int32 pdf_id = pdf_posterior[i][j].first;
              BaseFloat weight = pdf_posterior[i][j].second;
              file_like += map_accs.AccumulateForGmm(copy_am_gmm,
                                                     feats.Row(i),
                                                     pdf_id, weight);
              file_t += weight;
            }
          }

          KALDI_VLOG(2) << "Average like for utterance " << utt << " is "
                        << (file_like/file_t) << " over " << file_t << " frames.";

          tot_like += file_like;
          tot_t += file_t;
          num_done++;

          if (num_done % 10 == 0)
            KALDI_VLOG(1) << "Avg like per frame so far is "
                          << (tot_like / tot_t);
        }  // end looping over all utterances of the current speaker

        // MAP estimation.
        BaseFloat spk_objf_change = 0.0, spk_frames = 0.0;
        MapAmDiagGmmUpdate(map_config, map_accs, update_flags, &copy_am_gmm,
                           &spk_objf_change, &spk_frames);
        KALDI_LOG << "For speaker " << spk << ", objective function change "
                  << "from MAP was " << (spk_objf_change / spk_frames)
                  << " over " << spk_frames << " frames.";
        tot_like_change += spk_objf_change;
        tot_t_check += spk_frames;

        // Writing AM for each speaker in a table
        map_am_writer.Write(spk,copy_am_gmm);
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for ( ; !feature_reader.Done(); feature_reader.Next() ) {
        std::string utt = feature_reader.Key();
        AmDiagGmm copy_am_gmm;
        copy_am_gmm.CopyFromAmDiagGmm(am_gmm);
        AccumAmDiagGmm map_accs;
        map_accs.Init(am_gmm, update_flags);
        map_accs.SetZero(update_flags);

        if ( !posteriors_reader.HasKey(utt) ) {
          KALDI_WARN << "Did not find aligned transcription for utterance "
                     << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(utt);

        if ( posterior.size() != feats.NumRows() ) {
          KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
                     << " vs. " << (feats.NumRows());
          num_err++;
          continue;
        }
        num_done++;
        BaseFloat file_like = 0.0, file_t = 0.0;
        Posterior pdf_posterior;
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
        for ( size_t i = 0; i < posterior.size(); i++ ) {
          for ( size_t j = 0; j < pdf_posterior[i].size(); j++ ) {
            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat prob = pdf_posterior[i][j].second;
            file_like += map_accs.AccumulateForGmm(copy_am_gmm,feats.Row(i),
                                                   pdf_id, prob);
            file_t += prob;
          }
        }
        KALDI_VLOG(2) << "Average like for utterance " << utt << " is "
                      << (file_like/file_t) << " over " << file_t << " frames.";
        tot_like += file_like;
        tot_t += file_t;
        if ( num_done % 10 == 0 )
          KALDI_VLOG(1) << "Avg like per frame so far is "
                        << (tot_like / tot_t);

        // MAP
        BaseFloat utt_objf_change = 0.0, utt_frames = 0.0;
        MapAmDiagGmmUpdate(map_config, map_accs, update_flags, &copy_am_gmm,
                           &utt_objf_change, &utt_frames);
        KALDI_LOG << "For utterance " << utt << ", objective function change "
                  << "from MAP was " << (utt_objf_change / utt_frames)
                  << " over " << utt_frames << " frames.";
        tot_like_change += utt_objf_change;
        tot_t_check += utt_frames;

        // Writing AM for each utterance in a table
        map_am_writer.Write(feature_reader.Key(), copy_am_gmm);
      }
    }
    KALDI_ASSERT(ApproxEqual(tot_t, tot_t_check));
    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors";
    KALDI_LOG << "Overall acoustic likelihood was " << (tot_like / tot_t)
              << " and change in likelihod per frame was "
              << (tot_like_change / tot_t) << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
