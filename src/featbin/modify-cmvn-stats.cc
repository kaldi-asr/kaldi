// featbin/modify-cmvn-stats.cc

// Copyright      2014  Johns Hopkins University (author: Daniel Povey)

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
        "Copy cepstral mean/variance stats so that some dimensions have 'fake' stats\n"
        "that will skip normalization\n"
        "Copy features [and possibly change format]\n"
        "Usage: modify-cmvn-stats <fake-dims> <in-rspecifier> <out-wspecifier>\n"
        "e.g.: modify-cmvn-stats 13:14:15 ark:- ark:-\n"
        "See also: compute-cmvn-stats\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;

    std::string
        fake_dims_str = po.GetArg(1),
        rspecifier = po.GetArg(2),
        wspecifier = po.GetArg(3);

    std::vector<int32> fake_dims;
    if (!SplitStringToIntegers(fake_dims_str, ":", false, &fake_dims)) {
      KALDI_ERR << "Bad first argument (should be colon-separated list of "
                <<  "integers)";
    }
    
    SequentialDoubleMatrixReader reader(rspecifier);
    DoubleMatrixWriter writer(wspecifier);

    for (; !reader.Done(); reader.Next()) {
      Matrix<double> mat(reader.Value());

      if (mat.NumRows() != 2)
        KALDI_ERR << "Expected input to be CMVN stats (should have two rows)";

      int32 dim = mat.NumCols() - 1;
      double count = mat(0, dim);
      for (size_t i = 0; i < fake_dims.size(); i++) {
        int32 d = fake_dims[i];
        if (!(d >= 0 && d < dim))
          KALDI_ERR << "Bad entry " << d << " in list of fake dims; "
                    << "feature dim is " << dim;
        mat(0, d) = 0.0;  // zero 'x' stats.
        mat(1, d) = count;  // 'x^2' stats equalt to count, implying unit variance.
      }
      writer.Write(reader.Key(), mat);
      num_done++;
    }
    KALDI_LOG << "Modified " << num_done << " sets of stats.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


