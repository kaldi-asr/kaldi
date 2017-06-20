// nnetbin/nnet-kl-hmm-acc.cc

// Copyright 2013  Idiap Research Institute (Author: David Imseng)
//                 Karlsruhe Institute of Technology (Author: Ngoc Thang Vu)
//                 Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-kl-hmm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Collect the statistics for the Kl-HMM training.\n"
      "Usage: nnet-kl-hmm-acc [options] <feature-rspecifier> "
      "<alignments-rspecifier> <kl-hmm-accumulator>\n"
      "e.g.: nnet-kl-hmm-acc ark:feats.ark ark:ali.ark kl-hmm.acc\n";

    ParseOptions po(usage);

    bool binary = false;
    int32 n_kl_states = 0;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("nkl-states", &n_kl_states, "Number of states in Kl-HMM");


    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      alignments_rspecifier = po.GetArg(2),
      kl_hmm_accumulator = po.GetArg(3);

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    int32 posterior_dim = feature_reader.Value().NumCols();
    KlHmm kl_hmm(posterior_dim, n_kl_states);

    int32 num_done = 0,
          num_no_alignment = 0,
          num_other_error = 0;

    // main loop,
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!alignments_reader.HasKey(utt)) {
        num_no_alignment++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(utt);
        // Check,
        if (static_cast<int32>(alignment.size()) != mat.NumRows()) {
          KALDI_WARN << "Length mismatch! alignment " << alignment.size()
                     << ", feature-rows " << mat.NumRows()
                     << ", " << utt;
          num_other_error++;
          continue;
        }
        // Accumulate the statistics,
        kl_hmm.Accumulate(mat, alignment);
        KALDI_VLOG(2) << "utt " << utt << ", frames " << alignment.size();
        total_frames += mat.NumRows();
        num_done++;
      }
    }

    // Store the accumulator,
    {
      Output out(kl_hmm_accumulator, binary);
      kl_hmm.WriteData(out.Stream(), binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
