// bin/est-pca.cc

// Copyright      2015  Johns Hopkins University  (author: Sri Harish Mallidi)

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
#include "matrix/matrix-lib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate PCA statistics.\n"
        "Usage:  acc-pca [options] (<feature-rspecifier>|<vector-rspecifier>) <pca-acc-out>\n"
        "e.g.:\n"
        "  acc-pca scp:data/train/feats.scp pcaacc.1\n";

    bool binary = true;
    bool read_vectors = false;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Register("read-vectors", &read_vectors, "If true, read in single vectors "
                "instead of feature matrices");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1),
        pca_acc_wxfilename = po.GetArg(2);

    int32 num_done = 0, num_err = 0;
    int64 count = 0;
    Vector<double> sum;
    SpMatrix<double> sumsq;

    if (!read_vectors) {
      SequentialBaseFloatMatrixReader feat_reader(rspecifier);
    
      for (; !feat_reader.Done(); feat_reader.Next()) {
        Matrix<double> mat(feat_reader.Value());
        if (mat.NumRows() == 0) {
          KALDI_WARN << "Empty feature matrix";
          num_err++;
          continue;
        }
        if (sum.Dim() == 0) {
          sum.Resize(mat.NumCols());
          sumsq.Resize(mat.NumCols());
        }
        if (sum.Dim() != mat.NumCols()) {
          KALDI_WARN << "Feature dimension mismatch " << sum.Dim() << " vs. "
                     << mat.NumCols();
          num_err++;
          continue;
        }
        sum.AddRowSumMat(1.0, mat);
        sumsq.AddMat2(1.0, mat, kTrans, 1.0);
        count += mat.NumRows();
        num_done++;
      }
      KALDI_LOG << "Accumulated stats from " << num_done << " feature files, "
                << num_err << " with errors; " << count << " frames.";      
    } else {
      // read in vectors, not matrices
      SequentialBaseFloatVectorReader vec_reader(rspecifier);
    
      for (; !vec_reader.Done(); vec_reader.Next()) {
        Vector<double> vec(vec_reader.Value());
        if (vec.Dim() == 0) {
          KALDI_WARN << "Empty input vector";
          num_err++;
          continue;
        }
        if (sum.Dim() == 0) {
          sum.Resize(vec.Dim());
          sumsq.Resize(vec.Dim());
        }
        if (sum.Dim() != vec.Dim()) {
          KALDI_WARN << "Feature dimension mismatch " << sum.Dim() << " vs. "
                     << vec.Dim();
          num_err++;
          continue;
        }
        sum.AddVec(1.0, vec);
        sumsq.AddVec2(1.0, vec);
        count += 1.0;
        num_done++;
      }
      KALDI_LOG << "Accumulated stats from " << num_done << " vectors, "
                << num_err << " with errors.";
    }
    if (num_done == 0)
      KALDI_ERR << "No data accumulated.";

    Output ko(pca_acc_wxfilename, binary);
    
    WriteToken(ko.Stream(), binary, "<Count>");
    WriteBasicType(ko.Stream(), binary, count);
    
    WriteToken(ko.Stream(), binary, "<Sum>");
    sum.Write(ko.Stream(), binary);
    
    WriteToken(ko.Stream(), binary, "<SumSq>");
    sumsq.Write(ko.Stream(), binary);
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


