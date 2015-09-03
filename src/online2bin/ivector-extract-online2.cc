// online2bin/ivector-extract-online2.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
#include "online2/online-ivector-feature.h"
#include "thread/kaldi-task-sequence.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for utterances every --ivector-period frames, using a trained\n"
        "iVector extractor and features and Gaussian-level posteriors.  Similar to\n"
        "ivector-extract-online but uses the actual online decoder code to do it,\n"
        "and does everything in-memory instead of using multiple processes.\n"
        "Note: the value of the --use-most-recent-ivector config variable is ignored\n"
        "it's set to false.  The <spk2utt-rspecifier> is mandatory, to simplify the code;\n"
        "if you want to do it separately per utterance, just make it of the form\n"
        "<utterance-id> <utterance-id>.\n"
        "The iVectors are output as an archive of matrices, indexed by utterance-id;\n"
        "each row corresponds to an iVector.  If --repeat=true, outputs the whole matrix\n"
        "of iVectors, not just every (ivector-period)'th frame\n"
        "The input features are the raw, non-cepstral-mean-normalized features, e.g. MFCC.\n"
        "\n"
        "Usage:  ivector-extract-online2 [options] <spk2utt-rspecifier> <feature-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        "  ivector-extract-online2 --config=exp/nnet2_online/nnet_online/conf/ivector_extractor.conf \\\n"
        "    ark:data/train/spk2utt scp:data/train/feats.scp ark,t:ivectors.1.ark\n";
    
    ParseOptions po(usage);
    
    OnlineIvectorExtractionConfig ivector_config;
    ivector_config.Register(&po);

    g_num_threads = 8;
    bool repeat = false;
    
    po.Register("num-threads", &g_num_threads,
                "Number of threads to use for computing derived variables "
                "of iVector extractor, at process start-up.");
    po.Register("repeat", &repeat,
                "If true, output the same number of iVectors as input frames "
                "(including repeated data).");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string spk2utt_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        ivectors_wspecifier = po.GetArg(3);
    
    double tot_ubm_loglike = 0.0, tot_objf_impr = 0.0, tot_t = 0.0,
        tot_length = 0.0, tot_length_utt_end = 0.0;
    int32 num_done = 0, num_err = 0;
    
    ivector_config.use_most_recent_ivector = false;
    OnlineIvectorExtractionInfo ivector_info(ivector_config);
    
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter ivector_writer(ivectors_wspecifier);
    
    
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      OnlineIvectorExtractorAdaptationState adaptation_state(
          ivector_info);
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
        
        OnlineMatrixFeature matrix_feature(feats);

        OnlineIvectorFeature ivector_feature(ivector_info,
                                             &matrix_feature);
        
        ivector_feature.SetAdaptationState(adaptation_state);

        int32 T = feats.NumRows(),
            n = (repeat ? 1 : ivector_config.ivector_period),
            num_ivectors = (T + n - 1) / n;
        
        Matrix<BaseFloat> ivectors(num_ivectors,
                                   ivector_feature.Dim());
        
        for (int32 i = 0; i < num_ivectors; i++) {
          int32 t = i * n;
          SubVector<BaseFloat> ivector(ivectors, i);
          ivector_feature.GetFrame(t, &ivector);
        }
        // Update diagnostics.

        tot_ubm_loglike += T * ivector_feature.UbmLogLikePerFrame();
        tot_objf_impr += T * ivector_feature.ObjfImprPerFrame();
        tot_length_utt_end += T * ivectors.Row(num_ivectors - 1).Norm(2.0);
        for (int32 i = 0; i < num_ivectors; i++)
          tot_length += T * ivectors.Row(i).Norm(2.0) / num_ivectors;
        tot_t += T;
        KALDI_VLOG(2) << "For utterance " << utt << " of speaker " << spk
                      << ", UBM loglike/frame was "
                      << ivector_feature.UbmLogLikePerFrame()
                      << ", iVector length (at utterance end) was "
                      << ivectors.Row(num_ivectors-1).Norm(2.0)
                      << ", objf improvement/frame from iVector estimation was "
                      << ivector_feature.ObjfImprPerFrame();

        ivector_feature.GetAdaptationState(&adaptation_state);
        ivector_writer.Write(utt, ivectors);
        num_done++;
      }
    }

    KALDI_LOG << "Estimated iVectors for " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Average objective-function improvement was "
              << (tot_objf_impr / tot_t) << " per frame, over "
              << tot_t << " frames (weighted).";
    KALDI_LOG << "Average iVector length was " << (tot_length / tot_t)
              << " and at utterance-end was " << (tot_length_utt_end / tot_t)
              << ", over " << tot_t << " frames (weighted); "
              << " expected length is "
              << sqrt(ivector_info.extractor.IvectorDim());

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
