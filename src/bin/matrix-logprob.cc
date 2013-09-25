// bin/matrix-logprob.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute the log-prob of a particular (e.g.) pdf sequence, derived from a pdf alignment,\n"
        "given log-probs in a matrix.  The log-probs are computed over the whole data and printed\n"
        "as a logging message. Optionally also write out the original matrix.  This is for use in\n"
        "neural net discriminative training (for computing objective functions).\n"
        "\n"
        "Usage: matrix-logprob [options] <matrix-rspecifier> <pdf-ali-rspecifier> [<matrix-wspecifier1>]\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string matrix_rspecifier = po.GetArg(1),
        pdf_ali_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetOptArg(3);
    
    SequentialBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    RandomAccessInt32VectorReader pdf_ali_reader(pdf_ali_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    int64 tot_num_frames = 0;
    double tot_log_prob = 0;
    
    int32 num_done = 0, num_err = 0;
    for (; !matrix_reader.Done(); matrix_reader.Next()) {
      std::string key = matrix_reader.Key();
      const Matrix<BaseFloat> &logprob = matrix_reader.Value();
      if (pdf_ali_reader.HasKey(key)) {
        const std::vector<int32> &ali = pdf_ali_reader.Value(key);
        int32 num_frames = ali.size();
        if (num_frames != logprob.NumRows()) {
          KALDI_WARN << "Alignment has wrong size " << num_frames
                     << " vs. " << logprob.NumRows() << " for utterance " << key;
          num_err++;
          continue;
        }
        for (int32 t = 0; t < num_frames; t++) {
          int32 pdf_id = ali[t];
          if (pdf_id < 0 || pdf_id > logprob.NumCols()) // I'm letting this be an error.
            KALDI_ERR << "PDF id " << pdf_id << " out of range: "
                      << " max is " << logprob.NumCols();
          tot_log_prob += logprob(t, pdf_id);
        }
        tot_num_frames += num_frames;
        num_done++;
      } else {
        KALDI_WARN << "No alignment for key " << key;
        num_err++;
      }
      if (!matrix_wspecifier.empty()) {
        matrix_writer.Write(key, logprob);
      }
    }

    KALDI_LOG << "Average log-prob per frame is " << (tot_log_prob / tot_num_frames)
              << " over " << tot_num_frames << " frames.";
    KALDI_LOG << "Successfully processed " << num_done << " utterances, "
              << num_err << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


