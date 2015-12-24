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
        "Estimate PCA transform using stats obtained with acc-pca. \n"
        "Usage:     est-pca-acc [options] <pca-matrix-out> <pca-acc-1> <pca-acc-2> ...\n"
        "est-pca does stat accumulation and eigenvalue estimation in a single program.\n"
        "acc-pca+est-pca-acc does it in 2 seperate codes. Helpful for large datasets"
        "Estimate PCA transform; dimension reduction is optional (if not specified\n"
        "we don't reduce the dimension; if you specify --normalize-variance=true,\n"
        "we normalize the (centered) covariance of the features, and if you specify\n"
        "--normalize-mean=true the mean is also normalized.  So a variety of transform\n"
        "types are supported.  Because this type of transform does not need too much\n"
        "data to estimate robustly, we don't support separate accumulator files;\n"
        "this program reads in the features directly.  For large datasets you may\n"
        "want to subset the features (see example below)\n"
        "By default the program reads in matrices (e.g. features), but with\n"
        "--read-vectors=true, can read in vectors (e.g. iVectors).\n";

    bool binary = true;
    bool read_vectors = false;
    bool normalize_variance = false;
    bool normalize_mean = false;
    int32 dim = -1;
    std::string full_matrix_wxfilename;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Register("dim", &dim, "Feature dimension requested (if <= 0, uses full "
                "feature dimension");
    po.Register("read-vectors", &read_vectors, "If true, read in single vectors "
                "instead of feature matrices");
    po.Register("normalize-variance", &normalize_variance, "If true, make a "
                "transform that normalizes variance to one.");
    po.Register("normalize-mean", &normalize_mean, "If true, output an affine "
                "transform that subtracts the data mean.");
    po.Register("write-full-matrix", &full_matrix_wxfilename,
                "Write full version of the matrix to this location (including "
                "rejected rows)");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string pca_mat_wxfilename = po.GetArg(1);
    
    int64 count = 0;
    Vector<double> sum;
    SpMatrix<double> sumsq;

    for (int32 i = 2; i <= po.NumArgs(); i++) {
      bool binary_in;

      int64 this_count = 0;
      Vector<double> this_sum;
      SpMatrix<double> this_sumsq;

      Input ki(po.GetArg(i), &binary_in);

      ExpectToken(ki.Stream(), binary_in, "<Count>");
      ReadBasicType(ki.Stream(), binary_in, &this_count);
      count += this_count;

      ExpectToken(ki.Stream(), binary_in, "<Sum>");
      this_sum.Read(ki.Stream(), binary_in);

      if (sum.Dim() == 0) {
        sum.Resize(this_sum.Dim());
	sumsq.Resize(this_sum.Dim());
      }

      sum.AddVec(1.0, this_sum);

      ExpectToken(ki.Stream(), binary_in, "<SumSq>");
      this_sumsq.Read(ki.Stream(), binary_in);
      sumsq.AddSp(1.0, this_sumsq);

    }

    sum.Scale(1.0 / count);
    sumsq.Scale(1.0 / count);

    sumsq.AddVec2(-1.0, sum); // now sumsq is centered covariance.

    int32 full_dim = sum.Dim();
    if (dim <= 0) dim = full_dim;
    if (dim > full_dim)
      KALDI_ERR << "Final dimension " << dim << " is greater than feature "
                << "dimension " << full_dim;
    
    Matrix<double> P(full_dim, full_dim);
    Vector<double> s(full_dim);
    
    sumsq.Eig(&s, &P);
    SortSvd(&s, &P);
    
    KALDI_LOG << "Eigenvalues in PCA are " << s;
    KALDI_LOG << "Sum of PCA eigenvalues is " << s.Sum() << ", sum of kept "
              << "eigenvalues is " << s.Range(0, dim).Sum();


    Matrix<double> transform(P, kTrans); // Transpose of P.  This is what
                                         // appears in the transform.

    if (normalize_variance) {
      for (int32 i = 0; i < full_dim; i++) {
        double this_var = s(i), min_var = 1.0e-15;
        if (this_var < min_var) {
          KALDI_WARN << "--normalize-variance option: very tiny variance " << s(i)
                     << "encountered, treating as " << min_var;
          this_var = min_var;
        }
        double scale = 1.0 / sqrt(this_var); // scale on features that will make
                                             // the variance unit.
        transform.Row(i).Scale(scale);
      }
    }

    Vector<double> offset(full_dim);
    
    if (normalize_mean) {
      offset.AddMatVec(-1.0, transform, kNoTrans, sum, 0.0);
      transform.Resize(full_dim, full_dim + 1, kCopyData); // Add column to transform.
      transform.CopyColFromVec(offset, full_dim);
    }

    Matrix<BaseFloat> transform_float(transform);

    if (full_matrix_wxfilename != "") {
      WriteKaldiObject(transform_float, full_matrix_wxfilename, binary);
    }

    transform_float.Resize(dim, transform_float.NumCols(), kCopyData);
    WriteKaldiObject(transform_float, pca_mat_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


