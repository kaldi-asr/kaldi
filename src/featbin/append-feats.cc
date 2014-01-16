// featbin/append-feats.cc

// Copyright 2012   Petr Motlicek  Pawel Swietojanski
//                  Johns Hopkins University (author: Daniel Povey)

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
        "Append 2 feature-streams [and possibly change format]\n"
        "Note, this is deprecated; please use paste-feats\n"
        "Usage: append-feats [options] <in-rspecifier1> <in-rspecifier2> <out-wspecifier>\n"
        "\n"
        "e.g.: append-feats --feats-offset-in1 5 --num-feats-in1 5 scp:list1.scp "
        "scp:list2.scp ark:-\n";

    ParseOptions po(usage);

    bool truncate_frames = false;
    
    po.Register("truncate-frames", &truncate_frames, "If true, do not treat it "
                "as an error when files differ in number of frames, but truncate "
                "the longest one.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier1 = po.GetArg(1);
    std::string rspecifier2 = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feats_writer(wspecifier);
    SequentialBaseFloatMatrixReader feats_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feats_reader2(rspecifier2);

    int32 num_done = 0, num_err = 0;

    for (; !feats_reader1.Done(); feats_reader1.Next()) {
      std::string utt = feats_reader1.Key();
      if (!feats_reader2.HasKey(utt)) {
        KALDI_WARN << "Could not find features for " << utt << " in "
                   << rspecifier2 << ": producing no output for the utterance";
        num_err++;
        continue;
      }
      
      const Matrix<BaseFloat> &feats1 = feats_reader1.Value();
      const Matrix<BaseFloat> &feats2 = feats_reader2.Value(utt);
      if (feats1.NumRows() != feats2.NumRows() && !truncate_frames) {
        KALDI_WARN << "For utterance " << utt << ", features have different "
                   << "#frames " << feats1.NumRows() << " vs. "
                   << feats2.NumRows() << ", producing no output (use "
                   << "--truncate-frames=true if you want output)";
        num_err++;
        continue;
      }
      int32 num_frames = std::min(feats1.NumRows(), feats2.NumRows()),
          dim1 = feats1.NumCols(), dim2 = feats2.NumCols();
      Matrix<BaseFloat> output(num_frames, dim1 + dim2, kUndefined);
      output.Range(0, num_frames, 0, dim1).CopyFromMat(
          feats1.Range(0, num_frames, 0, dim1));
      output.Range(0, num_frames, dim1, dim2).CopyFromMat(
          feats2.Range(0, num_frames, 0, dim2));
      
      feats_writer.Write(utt, output);
      num_done++;
    }
    KALDI_LOG << "Appended " << num_done << " feats; " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
