// featbin/compute-snr-targets.cc

// Copyright 2015   Vimal Manohar

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute snr targets using clean and noisy speech features\n"
        "Usage: compute-snr-targets [options] <clean-feature-rspecifier> <noisy-feature-rspecifier> <targets-wspecifier>\n"
        "e.g.: compute-snr-targets scp:clean.scp scp:noisy.scp ark:targets.ark\n";

    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0, num_success = 0;
    
    // Copying tables of features.
    std::string clean_rspecifier = po.GetArg(1),    
                noisy_rspecifier = po.GetArg(2),
                targets_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader noisy_reader(noisy_rspecifier);
    RandomAccessBaseFloatMatrixReader clean_reader(clean_rspecifier);
    BaseFloatMatrixWriter kaldi_writer(targets_wspecifier);

    for (; !noisy_reader.Done(); noisy_reader.Next(), num_done++) {
      std::string key = noisy_reader.Key();
      const Matrix<BaseFloat> &noisy_feats = noisy_reader.Value();

      std::string uniq_key = key;
      if (!clean_reader.HasKey(uniq_key)) {
        KALDI_WARN << "Could not find uniq key " << uniq_key << " "
                   << "in clean feats " << clean_rspecifier;
        num_err++;
        continue;
      }

      Matrix<BaseFloat> clean_feats = clean_reader.Value(uniq_key);
      clean_feats.AddMat(-1.0, noisy_feats);
      kaldi_writer.Write(key, clean_feats);
      num_success++;
    }
    KALDI_LOG << "Computed SNR targets for " << num_success 
              << " out of " << num_done << " utterances; failed for "
              << num_err;
    return (num_success > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
