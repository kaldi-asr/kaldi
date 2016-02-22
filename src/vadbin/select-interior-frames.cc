// ivectorbin/select-interior-frames.cc

// Copyright   2013   Daniel Povey
//             2015   Vimal Manohar

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
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Select a subset of frames of the input files, based on the output of\n"
        "compute-vad or a similar program (a vector of length num-frames,\n"
        "containing 1.0 for voiced, 0.0 for unvoiced).\n"
        "Usage: select-voiced-frames [options] <feats-rspecifier> "
        " <vad-rspecifier> <feats-wspecifier>\n"
        "E.g.: select-voiced-frames [options] scp:feats.scp scp:vad.scp ark:-\n";
 
    bool select_unvoiced_frames = false;
    int32 padding = 0;

    ParseOptions po(usage);
    po.Register("select-unvoiced-frames", &select_unvoiced_frames, 
                "Reverses the operation of this file and selects "
                "unvoiced frames instead");
    po.Register("padding", &padding, 
                "Ignore frames at a boundary of this many frames");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string feat_rspecifier = po.GetArg(1),
        vad_rspecifier = po.GetArg(2),
        feat_wspecifier = po.GetArg(3);
    
    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessInt32VectorReader vad_reader(vad_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int32 num_done = 0, num_err = 0;
    long long num_frames = 0, num_select = 0;
    
    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      if (!vad_reader.HasKey(utt)) {
        KALDI_WARN << "No VAD input found for utterance " << utt;
        num_err++;
        continue;
      }
      const std::vector<int32> &voiced = vad_reader.Value(utt);

      if (std::abs(static_cast<int32>(feat.NumRows()) - static_cast<int32>(voiced.size())) > 1) {
        KALDI_WARN << "Mismatch in number for frames " << feat.NumRows() 
                   << " for features and VAD " << voiced.size() 
                   << ", for utterance " << utt;
        num_err++;
        continue;
      }
      int32 dim = 0;
      for (std::vector<int32>::const_iterator it = voiced.begin();
            it != voiced.end(); ++it) {
        if (!select_unvoiced_frames) {
          if (*it != 0)
            dim++;
        } else {
          if (*it == 0)
            dim++;
        }
      }

      if (dim == 0) {
        if (select_unvoiced_frames) {
          KALDI_WARN << "No unvoiced frames found for utterance " << utt;
        } else {
          KALDI_WARN << "No voiced frames found in utterance " << utt;
        }
        num_err++;
        continue;
      }
      Matrix<BaseFloat> voiced_feat(dim, feat.NumCols());
      int32 index = 0; 
      bool voiced_state = false;
      int32 start_idx = 0, end_idx = 0;
      for (int32 i = 0; i < std::min(static_cast<int32>(feat.NumRows()),static_cast<int32>(voiced.size())); i++) {
        if ((!voiced_state && voiced[i] != 0) || (voiced_state && voiced[i] == 0)) {
          // Reached voiced state from unvoiced state
          // or unvoiced state from voiced state
          end_idx = i;
          if ((!voiced_state && select_unvoiced_frames) || (voiced_state && !select_unvoiced_frames)) {
            if (end_idx - start_idx > 2 * padding && start_idx + padding < feat.NumRows()) {
              KALDI_ASSERT(index < voiced_feat.NumRows() && index + end_idx - start_idx - 2 * padding <= voiced_feat.NumRows());
              SubMatrix<BaseFloat> src_feat(feat, start_idx + padding, end_idx - start_idx - 2 * padding, 0, feat.NumCols());
              SubMatrix<BaseFloat> dst_feat(voiced_feat, index, end_idx - start_idx - 2 * padding, 0, feat.NumCols());
              dst_feat.CopyFromMat(src_feat);
              index += end_idx - start_idx - 2 * padding;
            }
          }
          start_idx = i;
          voiced_state = !voiced_state;
        }
      }

      if (!voiced_state && select_unvoiced_frames) {
        end_idx = std::min(static_cast<int32>(feat.NumRows()),static_cast<int32>(voiced.size()));
        if (end_idx - start_idx > 2 * padding && start_idx + padding < feat.NumRows()) {
          KALDI_ASSERT(index < voiced_feat.NumRows() && index + end_idx - start_idx - 2 * padding <= voiced_feat.NumRows());
          SubMatrix<BaseFloat> src_feat(feat, start_idx + padding, end_idx - start_idx - 2 * padding, 0, feat.NumCols());
          SubMatrix<BaseFloat> dst_feat(voiced_feat, index, end_idx - start_idx - 2 * padding, 0, feat.NumCols());
          dst_feat.CopyFromMat(src_feat);
        }
      }

      feat_writer.Write(utt, voiced_feat);
      num_select += voiced_feat.NumRows();
      num_frames += feat.NumRows();

      num_done++;
    }

    KALDI_LOG << "Done selecting " << num_select << " voiced frames"
              << " out of " << num_frames << " frames; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


