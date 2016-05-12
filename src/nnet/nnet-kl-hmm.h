// nnet/nnet-kl-hmm.h

// Copyright 2013  Idiap Research Institute (Author: David Imseng)
//                 Karlsruhe Institute of Technology (Author: Ngoc Thang Vu)
//                 Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_KL_HMM_H_
#define KALDI_NNET_NNET_KL_HMM_H_

#include <vector>

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {
namespace nnet1 {

class KlHmm : public Component {
 public:
  KlHmm(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out),
    kl_stats_(dim_out, dim_in, kSetZero)
  { }

  ~KlHmm()
  { }

  Component* Copy() const { return new KlHmm(*this); }
  ComponentType GetType() const { return kKlHmm; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    if (kl_inv_q_.NumRows() == 0) {
      // Copy the CudaMatrix to a Matrix
      Matrix<BaseFloat> in_tmp(in.NumRows(), in.NumCols());
      in.CopyToMat(&in_tmp);
      // Check if there are posteriors in the Matrix (check on first row),
      BaseFloat post_sum = in_tmp.Row(0).Sum();
      KALDI_ASSERT(ApproxEqual(post_sum, 1.0));
      // Get a tmp Matrix of the stats
      Matrix<BaseFloat> kl_stats_tmp(kl_stats_);
      // Init a vector to get the sum of the rows (for normalization)
      Vector<BaseFloat> row_sum(kl_stats_.NumRows(), kSetZero);
      // Get the sum of the posteriors for normalization
      row_sum.AddColSumMat(1, kl_stats_tmp);
      // Apply floor to make sure there is no zero
      row_sum.ApplyFloor(1e-20);
      // Invert the sum (to normalize)
      row_sum.InvertElements();
      // Normalizing the statistics vector
      kl_stats_tmp.MulRowsVec(row_sum);
      // Apply floor before inversion and logarithm
      kl_stats_tmp.ApplyFloor(1e-20);
      // Apply invesion
      kl_stats_tmp.InvertElements();
      // Apply logarithm
      kl_stats_tmp.ApplyLog();
      // Inverted and logged values
      kl_inv_q_.Resize(kl_stats_.NumRows(), kl_stats_.NumCols());
      // Holds now log (1/Q)
      kl_inv_q_.CopyFromMat(kl_stats_tmp);
    }
    // Get the logarithm of the features for the Entropy calculation
    // Copy the CudaMatrix to a Matrix
    Matrix<BaseFloat> in_log_tmp(in.NumRows(), in.NumCols());
    in.CopyToMat(&in_log_tmp);
    // Flooring and log
    in_log_tmp.ApplyFloor(1e-20);
    in_log_tmp.ApplyLog();
    CuMatrix<BaseFloat> log_in(in.NumRows(), in.NumCols());
    log_in.CopyFromMat(in_log_tmp);
    // P*logP
    CuMatrix<BaseFloat> tmp_entropy(in);
    tmp_entropy.MulElements(log_in);
    // Getting the entropy (sum P*logP)
    CuVector<BaseFloat> in_entropy(in.NumRows(), kSetZero);
    in_entropy.AddColSumMat(1, tmp_entropy);
    // sum P*log (1/Q)
    out->AddMatMat(1, in, kNoTrans, kl_inv_q_, kTrans, 0);
    // (sum P*logP) + (sum P*log(1/Q)
    out->AddVecToCols(1, in_entropy);
    // return the negative KL-divergence
    out->Scale(-1);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    KALDI_ERR << "Unimplemented";
  }

  /// Reads the component content
  void ReadData(std::istream &is, bool binary) {
    kl_stats_.Read(is, binary);
    KALDI_ASSERT(kl_stats_.NumRows() == output_dim_);
    KALDI_ASSERT(kl_stats_.NumCols() == input_dim_);
  }

  /// Writes the component content
  void WriteData(std::ostream &os, bool binary) const {
    kl_stats_.Write(os, binary);
  }

  /// Set the statistics matrix
  void SetStats(const Matrix<BaseFloat> mat) {
    KALDI_ASSERT(mat.NumRows() == output_dim_);
    KALDI_ASSERT(mat.NumCols() == input_dim_);
    kl_stats_.Resize(mat.NumRows(), mat.NumCols());
    kl_stats_.CopyFromMat(mat);
  }

  /// Accumulate the statistics for KL-HMM paramter estimation,
  void Accumulate(const Matrix<BaseFloat> &posteriors,
                  const std::vector<int32> &alignment) {
    KALDI_ASSERT(posteriors.NumRows() == alignment.size());
    KALDI_ASSERT(posteriors.NumCols() == kl_stats_.NumCols());
    int32 num_frames = alignment.size();
    for (int32 i = 0; i < num_frames; i++) {
      // Casting float posterior to double (fixing numerical issue),
      Vector<double> temp(posteriors.Row(i));
      // Sum the postiors grouped by states from the alignment,
      kl_stats_.Row(alignment[i]).AddVec(1, temp);
    }
  }

 private:
  Matrix<double> kl_stats_;
  CuMatrix<BaseFloat> kl_inv_q_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_KL_HMM_H_

