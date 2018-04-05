// featbin/concat-feats.cc

// Copyright 2013 Johns Hopkins University (Author: Daniel Povey)
//           2015 Tom Ko

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

namespace kaldi {

/*
   This function concatenates several sets of feature vectors
   to form a longer set. The length of the output will be equal
   to the sum of lengths of the inputs but the dimension will be
   the same to the inputs.
*/

void ConcatFeats(const std::vector<Matrix<BaseFloat> > &in,
                 Matrix<BaseFloat> *out) {
  KALDI_ASSERT(in.size() >= 1);
  int32 tot_len = in[0].NumRows(),
      dim = in[0].NumCols();
  for (int32 i = 1; i < in.size(); i++) {
    KALDI_ASSERT(in[i].NumCols() == dim);
    tot_len += in[i].NumRows();
  }
  out->Resize(tot_len, dim);
  int32 len_offset = 0;
  for (int32 i = 0; i < in.size(); i++) {
    int32 this_len = in[i].NumRows();
    out->Range(len_offset, this_len, 0, dim).CopyFromMat(
        in[i]);
    len_offset += this_len;
  }
}


}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Concatenate feature files (assuming they have the same dimensions),\n"
        "so the output file has the sum of the num-frames of the inputs.\n"
        "Usage: concat-feats <in-rxfilename1> <in-rxfilename2> [<in-rxfilename3> ...] <out-wxfilename>\n"
        " e.g. concat-feats mfcc/foo.ark:12343 mfcc/foo.ark:56789 -\n"
        "See also: copy-feats, append-vector-to-feats, paste-feats\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<Matrix<BaseFloat> > feats(po.NumArgs() - 1);
    for (int32 i = 1; i < po.NumArgs(); i++)
      ReadKaldiObject(po.GetArg(i), &(feats[i-1]));
    Matrix<BaseFloat> output;
    ConcatFeats(feats, &output);
    std::string output_wxfilename = po.GetArg(po.NumArgs());
    WriteKaldiObject(output, output_wxfilename, binary);

    // This will tend to produce too much output if we have a logging mesage.
    // KALDI_LOG << "Wrote concatenated features to " << output_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
