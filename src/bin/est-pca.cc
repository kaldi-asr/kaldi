// bin/est-pca.cc

// Copyright      2014  Johns Hopkins University  (author: Daniel Povey)

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
        "Estimate PCA transform; dimension reduction is optional (if not specified\n"
        "we don't reduce the dimension; if you specify --normalize-variance=true,\n"
        "we normalize the (centered) covariance of the features, and if you specify\n"
        "--normalize-mean=true the mean is also normalized.  So a variety of transform\n"
        "types are supported.  Because this type of transform does not need too much\n"
        "data to estimate robustly, we don't support separate accumulator files;\n"
        "this program reads in the features directly.  For large datasets you may\n"
        "want to subset the features (see example below)\n"
        "By default the program reads in matrices (e.g. features), but with\n"
        "--read-vectors=true, can read in vectors (e.g. iVectors).\n"
        "\n"
        "Usage:  est-pca [options] (<feature-rspecifier>|<vector-rspecifier>) <pca-matrix-out>\n"
        "e.g.:\n"
        "utils/shuffle_list.pl data/train/feats.scp | head -n 5000 | sort | \\\n"
        "  est-pca --dim=50 scp:- some/dir/0.mat\n";

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

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1),
        pca_mat_wxfilename = po.GetArg(2);

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


