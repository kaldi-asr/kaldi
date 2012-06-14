// featbin/append-feats.cc

// Copyright 2012   Petr Motlicek;  Pawel Swietojanski

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
        "Usage: append-feats [options] in-rspecifier1 in-rspecifier2 out-wspecifier\n"
        "Example: append-feats --feats-offset-in1 5 --num-feats-in1 5 scp:list1.scp "
        "scp:list2.scp ark:-\n";

    ParseOptions po(usage);

    int32 feats_offset_in1 = 0;
    int32 feats_offset_in2 = 0;
    int32 num_feats_in1 = 0;
    int32 num_feats_in2 = 0;

    po.Register("feats-offset-in1", &feats_offset_in1, "Feats 1 offset");
    po.Register("num-feats-in1", &num_feats_in1, "Take num-feats from in1-rspeciifier");
    po.Register("feats-offset-in2", &feats_offset_in2, "Feats 2 offset");
    po.Register("num-feats-in2", &num_feats_in2, "Take num-feats from in2-rspeciifier");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier1 = po.GetArg(1);
    std::string rspecifier2 = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    KALDI_ASSERT(feats_offset_in1 >= 0 && feats_offset_in2 >= 0);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader kaldi_reader2(rspecifier2);

    // Peeking in the archives to get the feature dimensions
    if (kaldi_reader1.Done()) {
      KALDI_ERR << "Could not read any features from " << rspecifier1
                << ". (empty archive?)";
    }
    std::string utt = kaldi_reader1.Key();
    if (!kaldi_reader2.HasKey(utt)) {
      KALDI_ERR << "Could not read features for key " << utt << " from "
                << rspecifier2 << ". (empty archive?)";
    }

    int32 dim_feats_in1 = kaldi_reader1.Value().NumCols();
    int32 dim_feats_in2 = kaldi_reader2.Value(utt).NumCols();
    if (num_feats_in1 == 0)
      num_feats_in1 = dim_feats_in1 - feats_offset_in1;
    if (num_feats_in2 == 0)
      num_feats_in2 = dim_feats_in2 - feats_offset_in2;

    KALDI_LOG << "Reading features from " << rspecifier1 << " and " << rspecifier2;
    KALDI_LOG << "\tdim1 = " << dim_feats_in1 << "; offset1 = " << feats_offset_in1
              << "; num1 = " << num_feats_in1 << "; dim2 = " << dim_feats_in2
              << "; offset2 = " << feats_offset_in2 << "; num2 = " << num_feats_in2;

    KALDI_ASSERT((feats_offset_in1 + num_feats_in1) <= dim_feats_in1);
    KALDI_ASSERT((feats_offset_in2 + num_feats_in2) <= dim_feats_in2);

    for (; !kaldi_reader1.Done(); kaldi_reader1.Next()) {
      utt = kaldi_reader1.Key();
      if (!kaldi_reader2.HasKey(utt)) {
        KALDI_WARN << "Could not find features for " << utt << " in "
                   << rspecifier2 << ": producing no output for the utterance";
        continue;
      }

      const Matrix<BaseFloat> &feats1 = kaldi_reader1.Value();
      const Matrix<BaseFloat> &feats2 = kaldi_reader2.Value(utt);
      int32 num_frames = feats1.NumRows();
      KALDI_VLOG(1) << "Utterance : " << utt << ": # of frames = " << num_frames;

      KALDI_ASSERT(feats1.NumCols() == dim_feats_in1 &&
                   feats2.NumCols() == dim_feats_in2);
      if (num_frames != feats2.NumRows()) {
        KALDI_WARN << "Utterance " << utt << ": " << num_frames
                   << " frames read from " << rspecifier1 << " and "
                   << feats2.NumRows() << " frames read from " << rspecifier2
                   << ": producing no output for the utterance";
        continue;
      }

      SubMatrix<BaseFloat> new_feats1(feats1, 0, num_frames, feats_offset_in1,
                                      num_feats_in1);
      SubMatrix<BaseFloat> new_feats2(feats2, 0, num_frames, feats_offset_in2,
                                      num_feats_in2);
      Matrix<BaseFloat> output_feats(num_frames, new_feats1.NumCols() +
                                     new_feats2.NumCols());
      output_feats.Range(0, num_frames, 0,
                         new_feats1.NumCols()).CopyFromMat(new_feats1);
      output_feats.Range(0, num_frames, new_feats1.NumCols(),
                         new_feats2.NumCols()).CopyFromMat(new_feats2);
      kaldi_writer.Write(utt, output_feats);
    }

    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


