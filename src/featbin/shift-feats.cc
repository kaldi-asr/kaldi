// featbin/shift-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//           2013-2015  Johns Hopkins University (author: Daniel Povey)

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
        "Copy features and possibly shift them in time while maintaining the length, e.g.\n"
        "shift-feats --shift=1 <input-feats> <output-feats> will shift all frames to the\n"
        "right by one (the first frame would be duplicated).\n"
        "See also: copy-feats, copy-matrix\n";

    ParseOptions po(usage);
    int32 shift = 0;
    po.Register("shift", &shift, "Number of frames by which to shift the features.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0;

    SequentialBaseFloatMatrixReader feat_reader(po.GetArg(1));
    BaseFloatMatrixWriter feat_writer(po.GetArg(2));


    for (; !feat_reader.Done(); feat_reader.Next()) {
      const std::string &key = feat_reader.Key();
      const Matrix<BaseFloat> &src = feat_reader.Value();
      if (src.NumRows() == 0) {
        KALDI_WARN << "Empty matrix for key " << key;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> rearranged(src.NumRows(), src.NumCols());
      for (int32 r = 0; r < src.NumRows(); r++) {
        int32 src_r = r - shift;
        if (src_r < 0) src_r = 0;
        if (src_r >= src.NumRows()) src_r = src.NumRows() - 1;
        rearranged.Row(r).CopyFromVec(src.Row(src_r));
      }
      feat_writer.Write(key, rearranged);
      num_done++;
    }

    KALDI_LOG << "Shifted " << num_done << " features by "
              << shift << " frames; " << num_err << " with errors.";
    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


/*
test:
  echo "foo [ 1 1; 2 2; 3 3 ]" | shift-feats --shift=1 ark:- ark,t:-
  outputs:
  foo  [
  1 1
  1 1
  2 2 ]
*/
