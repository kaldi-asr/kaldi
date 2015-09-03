// bin/build-pfile-from-ali.cc

// Copyright 2013  Carnegie Mellon University (Author: Yajie Miao)
//                 Johns Hopkins University (Author: Daniel Povey)

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
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"

/** @brief Build pfiles for Neural Network training from alignment.
 * The pfiles contains both the data vectors and their corresponding
 * class/state labels (zero-based).
*/

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Build pfiles for neural network training from alignment.\n"
        "Usage:  build-pfile-from-ali [options] <model> <alignments-rspecifier> <feature-rspecifier> \n"
        "<pfile-wspecifier>\n"
        "e.g.: \n"
        " build-pfile-from-ali 1.mdl ark:1.ali features \n"
        " \"|pfile_create -i - -o pfile.1 -f 143 -l 1\" ";

    ParseOptions po(usage);

    int32 every_nth_frame = 1;
    po.Register("every-nth-frame", &every_nth_frame, "This option will cause it to print "
                "out only every n'th frame (for subsampling)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        pfile_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);

    int32 num_done = 0, num_no_ali = 0, num_other_error = 0;
    int32 num_utt = 0;

    KALDI_ASSERT(every_nth_frame >= 1);
    
    Output ko(pfile_wspecifier, false);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!ali_reader.HasKey(key)) {
        KALDI_WARN << "Did not find alignment for utterance " << key;
        num_no_ali++;
        continue;
      }

      const Matrix<BaseFloat> &feats = feature_reader.Value();
      std::vector<int32> alignment = ali_reader.Value(key);
      if (static_cast<int32>(feats.NumRows()) != static_cast<int32>(alignment.size())) {
        KALDI_WARN << "Alignment vector has wrong size " << (alignment.size())
                   << " vs. " << (feats.NumRows());
        num_other_error++;
        continue;
      }
      int32 dim = feats.NumCols();

      for (size_t i = 0; i < alignment.size(); i++) {
        if (i % every_nth_frame == 0) {
          std::stringstream ss;
          // Output sentence number and frame number
          ss << num_utt;
          ss << " ";
          ss << (i / every_nth_frame);
          // Output feature vector
          for (int32 d = 0; d < dim; ++d) {
            ss << " ";
            ss << feats(i, d);
          }
          // Output the class label
          ss << " ";
          ss << trans_model.TransitionIdToPdf(alignment[i]);

          ko.Stream() << ss.str().c_str();
          ko.Stream() << "\n";
        }
      }
      num_done ++; num_utt ++;
    }
    ko.Close();
    KALDI_LOG << "Converted " << num_done << " alignments to pfiles.";
    KALDI_LOG << num_no_ali << " utterances have no alignment; "
              << num_other_error << " utterances have other errors.";
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


