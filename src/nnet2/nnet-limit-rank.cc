// nnet2/nnet-limit-rank.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-limit-rank.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi {
namespace nnet2 {

class LimitRankClass {
 public:
  LimitRankClass(const NnetLimitRankOpts &opts,
                 int32 c,
                 Nnet *nnet): opts_(opts), c_(c), nnet_(nnet) { }
  void operator () () {
    AffineComponent *ac = dynamic_cast<AffineComponent*>(
        &(nnet_->GetComponent(c_)));
    KALDI_ASSERT(ac != NULL);

    // We'll limit the rank of just the linear part, keeping the bias vector full.
    Matrix<BaseFloat> M (ac->LinearParams());
    int32 rows = M.NumRows(), cols = M.NumCols(), rc_min = std::min(rows, cols);
    Vector<BaseFloat> s(rc_min);
    Matrix<BaseFloat> U(rows, rc_min), Vt(rc_min, cols);
    // Do the destructive svd M = U diag(s) V^T.  It actually outputs the transpose of V.
    M.DestructiveSvd(&s, &U, &Vt);
    SortSvd(&s, &U, &Vt); // Sort the singular values from largest to smallest.

    int32 d = GetRetainedDim(rows, cols);
    BaseFloat old_svd_sum = s.Sum();
    U.Resize(rows, d, kCopyData);
    s.Resize(d, kCopyData);
    Vt.Resize(d, cols, kCopyData);
    BaseFloat new_svd_sum = s.Sum();
    KALDI_LOG << "For component " << c_ << " of dimension " << rows
              << " x " << cols << ", reduced rank from "
              << rc_min <<  " to " << d << ", SVD sum reduced from "
              << old_svd_sum << " to " << new_svd_sum;
    Vt.MulRowsVec(s); // Vt <-- diag(s) Vt.
    M.AddMatMat(1.0, U, kNoTrans, Vt, kNoTrans, 0.0); // Reconstruct with reduced
    // rank.
    Vector<BaseFloat> bias_params(ac->BiasParams());
    ac->SetParams(bias_params, M);
  }

  int32 GetRetainedDim(int32 rows, int32 cols) {
    if (opts_.parameter_proportion <= 0.0 || opts_.parameter_proportion > 1.0)
      KALDI_ERR << "bad --parameter-proportion " << opts_.parameter_proportion;
    // If we do SVD to dimension d, so that it's U diag(s) V^T where
    // U is rows * d, s is d, and V is cols * d, then the #params is as follows...
    //   the first column of U has free parameters (#rows - 1) [the -1 is due to
    //   the length constraint]; the second has (#rows - 2) [subtract 1 for the
    //   length constraint and one for orthogonality with the previous row], etc.
    //   Total is params(U) = (rows * d) - ((d(d+1))/2),
    //            params(s) = d,
    //            params(V) = (cols * d) - ((d(d+1))/2),
    //   So total is (rows + cols) * d - d * d .
    //   For example, if d = #rows, this equals (#rows * #cols)
    //   We are solving for:
    //   (rows * cols) * parameter_proportion = (rows + cols) * d - d * d, or
    //   d^2 - d * (rows + cols) + (rows*cols)*parameter_proportion
    //   In quadratic equation
    //   a = 1.0,
    //   b = -(rows + cols)
    //   c = rows * cols * parameter_proportion.
    //   Take smaller solution.
    BaseFloat a = 1.0, b = -(rows + cols),
        c = rows * cols * opts_.parameter_proportion;
    BaseFloat x = (-b - sqrt(b * b - 4 * a * c)) / (2.0 * a);
    int32 ans = static_cast<int32>(x);
    KALDI_ASSERT(ans > 0 && ans <= std::min(rows, cols));
    return ans;
  }
  
  ~LimitRankClass() { }
 private:
  const NnetLimitRankOpts &opts_;
  int32 c_;
  Nnet *nnet_;
};


void LimitRankParallel(const NnetLimitRankOpts &opts,
                            Nnet *nnet) {
  TaskSequencerConfig task_config;
  task_config.num_threads = opts.num_threads;
  TaskSequencer<LimitRankClass> tc(task_config);
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    if (dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c))) != NULL)
      tc.Run(new LimitRankClass(opts, c, nnet));
  }
}


} // namespace nnet2
} // namespace kaldi
