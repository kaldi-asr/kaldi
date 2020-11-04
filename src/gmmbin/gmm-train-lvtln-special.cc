// gmmbin/gmm-train-lvtln-special.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright 2014       Vimal Manohar

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
#include "transform/lvtln.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Set one of the transforms in lvtln to the minimum-squared-error solution\n"
        "to mapping feats-untransformed to feats-transformed; posteriors may\n"
        "optionally be used to downweight/remove silence.\n"
        "Usage: gmm-train-lvtln-special [options] class-index <lvtln-in> <lvtln-out> "
        " <feats-untransformed-rspecifier> <feats-transformed-rspecifier> [<posteriors-rspecifier>]\n"
        "e.g.: \n"
        " gmm-train-lvtln-special 5 5.lvtln 6.lvtln scp:train.scp scp:train_warp095.scp ark:nosil.post\n";

    BaseFloat warp = -1.0;
    bool binary = true;
    bool normalize_var = false;
    bool normalize_covar = false;
    std::string weights_rspecifier;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("warp", &warp, "If supplied, can be used to set warp factor "
                "for this transform");
    po.Register("normalize-var", &normalize_var, "Normalize diagonal of variance "
                "to be the same before and after transform.");
    po.Register("normalize-covar", &normalize_covar, "Normalize (matrix-valued) "
                "covariance to be the same before and after transform.");
    po.Register("weights-in", &weights_rspecifier, 
                "Can be used to take posteriors as an scp or ark file of weights "
                "instead of giving <posteriors-rspecfier>");

    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string class_idx_str = po.GetArg(1);
    int32 class_idx;
    if (!ConvertStringToInteger(class_idx_str, &class_idx))
      KALDI_ERR << "Expected integer first argument: got " << class_idx_str;

    std::string lvtln_rxfilename = po.GetArg(2),
        lvtln_wxfilename = po.GetArg(3),
        feats_orig_rspecifier = po.GetArg(4),
        feats_transformed_rspecifier = po.GetArg(5),
        posteriors_rspecifier = po.GetOptArg(6);

    // Get lvtln object.
    LinearVtln lvtln;
    ReadKaldiObject(lvtln_rxfilename, &lvtln);
    int32 dim = lvtln.Dim();  // feature dimension [we hope!].


    if (!normalize_covar) {
      // Below is the computation if we are not normalizing the full covariance.

      // Ignoring weighting (which is a straightforward extension), the problem is this:
      // we have original features x(t) and transformed features y(t) [both x(t) and y(t)
      // are vectors of size D].  We are training an affine transform to minimize the sum-squared
      // error between A x(t) + b and y(t).  Let x(t)^+ be x(t) with a 1 appended, and let
      // w_i be the i'th row of the matrix [ A; b ], as in CMLLR.
      //  We are minimizeing
      // \sum_{t = 1}^T \sum_{i = 1}^D   (w_i^T x(t)^+  - y_i(t))^2,
      // We can express this in terms of sufficient statistics as:
      //   \sum_{i = 1}^D   w_i^T Q w_i - 2 w_i^T l_i + c_i,
      // where
      //  Q_i = \sum_{t = 1}^T x(t)^+ x(t)^+^T
      //  l_i = \sum_{t = 1}^T x(t)^+ y_i(t)
      //  c_i = \sum_{t = 1}^T y_i(t)^2
      // The solution for row i is: w_i = Q^{-1} l_i
      //  and the sum-square error for index i is:
      //   w_i^T Q w_i - 2 w_i^T l_i + c_i .
      // Note that for lvtln purposes we throw away the "offset" element (i.e. the last
      // element of each row w_i).

      // Declare statistics we use to estimate transform.
      SpMatrix<double> Q(dim+1);  // quadratic stats == outer product of x^+.
      Matrix<double> l(dim, dim+1);  // i'th row of l is l_i
      Vector<double> c(dim);
      double beta = 0.0;
      Vector<double> sum_xplus(dim+1);  // sum of x_i^+
      Vector<double> sumsq_x(dim);  // sumsq of x_i
      Vector<double> sumsq_diff(dim);  // sum of x_i-y_i

      SequentialBaseFloatMatrixReader x_reader(feats_orig_rspecifier);
      RandomAccessBaseFloatMatrixReader y_reader(feats_transformed_rspecifier);

      RandomAccessPosteriorReader post_reader(posteriors_rspecifier);
      RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);

      for (; !x_reader.Done(); x_reader.Next()) {
        std::string utt = x_reader.Key();
        if (!y_reader.HasKey(utt)) {
          KALDI_WARN << "No transformed features for key " << utt;
          continue;
        }
        const Matrix<BaseFloat> &x_feats = x_reader.Value();
        const Matrix<BaseFloat> &y_feats = y_reader.Value(utt);
        if (x_feats.NumRows() != y_feats.NumRows() ||
           x_feats.NumCols() != y_feats.NumCols() ||
           x_feats.NumCols() != dim) {
          KALDI_ERR << "Number of rows and/or columns differs in features, or features have different dim from lvtln object";
        }

        Vector<BaseFloat> weights(x_feats.NumRows());
        if (weights_rspecifier == "" && posteriors_rspecifier != "") {
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "No posteriors for utterance " << utt;
            continue;
          }
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != x_feats.NumRows())
            KALDI_ERR << "Mismatch in size of posterior";
          for (size_t i = 0; i < post.size(); i++)
            for (size_t j = 0; j < post[i].size(); j++)
              weights(i) += post[i][j].second;
        } else if (weights_rspecifier != "") {
          if (!weights_reader.HasKey(utt)) {
            KALDI_WARN << "No weights for utterance " << utt;
            continue;
          }
          weights.CopyFromVec(weights_reader.Value(utt));
        } else {
          weights.Add(1.0);
        }

        // Now get stats.

        for (int32 i = 0; i < x_feats.NumRows(); i++) {
          BaseFloat weight = weights(i);
          SubVector<BaseFloat> x_row(x_feats, i);
          SubVector<BaseFloat> y_row(y_feats, i);
          Vector<double> xplus_row_dbl(dim+1);
          for (int32 j = 0; j < dim; j++)
            xplus_row_dbl(j) = x_row(j);
          xplus_row_dbl(dim) = 1.0;
          Vector<double> y_row_dbl(y_row);
          Q.AddVec2(weight, xplus_row_dbl);
          l.AddVecVec(weight, y_row_dbl, xplus_row_dbl);
          beta += weight;
          sum_xplus(dim) += weight;
          for (int32 j = 0; j < dim; j++) {
            sum_xplus(j) += weight * x_row(j);
            sumsq_x(j) += weight * x_row(j)*x_row(j);
            sumsq_diff(j) += weight * (x_row(j)-y_row(j)) * (x_row(j)-y_row(j));
            c(j) += weight * y_row(j)*y_row(j);
          }
        }
      }

      Matrix<BaseFloat> A(dim, dim);  // will give this to LVTLN object
      // as transform matrix.
      SpMatrix<double> Qinv(Q);
      Qinv.Invert();
      for (int32 i = 0; i < dim; i++) {
        Vector<double> w_i(dim+1);
        SubVector<double> l_i(l, i);
        w_i.AddSpVec(1.0, Qinv, l_i, 0.0);  // w_i = Q^{-1} l_i
        SubVector<double> a_i(w_i, 0, dim);
        A.Row(i).CopyFromVec(a_i);

        BaseFloat error = (VecSpVec(w_i, Q, w_i) - 2.0*VecVec(w_i, l_i) + c(i)) / beta,
            sqdiff = sumsq_diff(i) / beta,
            scatter = sumsq_x(i) / beta;

        KALDI_LOG << "For dimension " << i << ", sum-squared error in linear approximation is "
                  << error << ", versus feature-difference " << sqdiff << ", orig-sumsq is "
                  << scatter;
        if (normalize_var) {  // add a scaling to normalize the variance.
          double x_var = scatter - pow(sum_xplus(i) / beta, 2.0);
          double y_var = VecSpVec(w_i, Q, w_i)/beta
              - pow(VecVec(w_i, sum_xplus)/beta, 2.0);
          double scale = sqrt(x_var / y_var);
          KALDI_LOG << "For dimension " << i
                    << ", variance of original and transformed data is " << x_var
                    << " and " << y_var << " respectively; scaling matrix row by "
                    << scale << " to make them equal.";
          A.Row(i).Scale(scale);
        }
      }
      lvtln.SetTransform(class_idx, A);
    } else {
      // Here is the computation if we normalize the full covariance.
      // see the document "Notes for affine-transform-based VTLN" for explanation,
      // here: http://www.danielpovey.com/files/2010_vtln_notes.pdf
      
      double T = 0.0;
      SpMatrix<double> XX(dim);  // sum of x x^t
      Vector<double> x(dim);  //  sum of x.
      Vector<double> y(dim);  //  sum of y.
      Matrix<double> XY(dim, dim);  // sum of x y^t

      SequentialBaseFloatMatrixReader x_reader(feats_orig_rspecifier);
      RandomAccessBaseFloatMatrixReader y_reader(feats_transformed_rspecifier);

      RandomAccessPosteriorReader post_reader(posteriors_rspecifier);

      for (; !x_reader.Done(); x_reader.Next()) {
        std::string utt = x_reader.Key();
        if (!y_reader.HasKey(utt)) {
          KALDI_WARN << "No transformed features for key " << utt;
          continue;
        }
        const Matrix<BaseFloat> &x_feats = x_reader.Value();
        const Matrix<BaseFloat> &y_feats = y_reader.Value(utt);
        if (x_feats.NumRows() != y_feats.NumRows() ||
           x_feats.NumCols() != y_feats.NumCols() ||
           x_feats.NumCols() != dim) {
          KALDI_ERR << "Number of rows and/or columns differs in features, or features have different dim from lvtln object";
        }

        Vector<BaseFloat> weights(x_feats.NumRows());
        if (posteriors_rspecifier != "") {
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "No posteriors for utterance " << utt;
            continue;
          }
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != x_feats.NumRows())
            KALDI_ERR << "Mismatch in size of posterior";
          for (size_t i = 0; i < post.size(); i++)
            for (size_t j = 0; j < post[i].size(); j++)
              weights(i) += post[i][j].second;
        } else weights.Add(1.0);
        // Now get stats.
        for (int32 i = 0; i < x_feats.NumRows(); i++) {
          BaseFloat weight = weights(i);
          SubVector<BaseFloat> x_row(x_feats, i);
          SubVector<BaseFloat> y_row(y_feats, i);
          Vector<double> x_dbl(x_row);
          Vector<double> y_dbl(y_row);
          T += weight;
          XX.AddVec2(weight, x_dbl);
          x.AddVec(weight, x_row);
          y.AddVec(weight, y_row);
          XY.AddVecVec(weight, x_dbl, y_dbl);
        }
      }
      KALDI_ASSERT(T > 0.0);
      Vector<double> xbar(x); xbar.Scale(1.0/T);

      SpMatrix<double> S(XX); S.Scale(1.0/T);
      S.AddVec2(-1.0, xbar);
      TpMatrix<double> C_tp(dim);
      C_tp.Cholesky(S);  // get cholesky factor.
      TpMatrix<double> Cinv_tp(C_tp);
      Cinv_tp.Invert();
      Matrix<double> C(C_tp);  // use regular matrix as more stuff is implemented for this case.
      Matrix<double> Cinv(Cinv_tp);
      Matrix<double> P0(XY);
      P0.AddVecVec(-1.0, xbar, y);
      Matrix<double> P(dim, dim), tmp(dim, dim);
      tmp.AddMatMat(1.0, P0, kNoTrans, Cinv, kTrans, 0.0);  // tmp := P0 * C^{-T}
      P.AddMatMat(1.0, Cinv, kNoTrans, tmp, kNoTrans, 0.0);  // P := C^{-1} * P0
      Vector<double> l(dim);
      Matrix<double> U(dim, dim), Vt(dim, dim);
      P.Svd(&l, &U, &Vt);
      l.Scale(1.0/T);  //  normalize for diagnostic purposes.
      KALDI_LOG << "Singular values of P are: " << l;
      Matrix<double> N(dim, dim);
      N.AddMatMat(1.0, Vt, kTrans, U, kTrans, 0.0);  // N := V * U^T.
      Matrix<double> M(dim, dim);
      tmp.AddMatMat(1.0, N, kNoTrans, Cinv, kNoTrans, 0.0);  // tmp := N * C^{-1}
      M.AddMatMat(1.0, C, kNoTrans, tmp, kNoTrans, 0.0);  // M := C * tmp = C * N * C^{-1}
      Matrix<BaseFloat> Mf(M);
      lvtln.SetTransform(class_idx, Mf);  // in this setup we don't
      // need the offset, v.
    }

    if (warp >= 0.0)
      lvtln.SetWarp(class_idx, warp);

    {  // Write lvtln object.
      Output ko(lvtln_wxfilename, binary);
      lvtln.Write(ko.Stream(), binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

