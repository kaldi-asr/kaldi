// gmm/mle-diag-gmm.cc

// Copyright 2009-2011  Arnab Ghoshal

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

#include <algorithm>  // for std::max
#include <string>
#include <vector>

#include "gmm/diag-gmm.h"
#include "gmm/ebw-diag-gmm.h"

namespace kaldi {

void AccumEbwDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectMarker(in_stream, binary, "<GMMEBWACCS>");
  ExpectMarker(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectMarker(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  ExpectMarker(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if ((NumGauss() != 0 || Dim() != 0 || Flags() != 0)) {
      if (num_components != NumGauss() || dimension != Dim()
          || flags != Flags()) {
        KALDI_ERR << "Dimension or flags mismatch: " << NumGauss() << ", "
                  << Dim() << ", " << Flags() << " vs. " << num_components
                  << ", " << dimension << ", " << flags;
      }
    } else {
      Resize(num_components, dimension, flags);
    }
  } else {
    Resize(num_components, dimension, flags);
  }

  ReadMarker(in_stream, binary, &token);
  while (token != "</GMMEBWACCS>") {
    if (token == "<NUM_OCCUPANCY>") {
      num_occupancy_.Read(in_stream, binary, add);
    } else if (token == "<DEN_OCCUPANCY>") {
      den_occupancy_.Read(in_stream, binary, add);
    } else if (token == "<MEANACCS>") {
      mean_accumulator_.Read(in_stream, binary, add);
    } else if (token == "<DIAGVARACCS>") {
      variance_accumulator_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void AccumEbwDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<GMMEBWACCS>");
  WriteMarker(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteMarker(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> num_occupancy_bf(num_occupancy_.Dim());
  Vector<BaseFloat> den_occupancy_bf(den_occupancy_.Dim());
  Matrix<BaseFloat> mean_accumulator_bf(mean_accumulator_.NumRows(),
      mean_accumulator_.NumCols());
  Matrix<BaseFloat> variance_accumulator_bf(variance_accumulator_.NumRows(),
      variance_accumulator_.NumCols());
  num_occupancy_bf.CopyFromVec(num_occupancy_);
  den_occupancy_bf.CopyFromVec(den_occupancy_);
  mean_accumulator_bf.CopyFromMat(mean_accumulator_);
  variance_accumulator_bf.CopyFromMat(variance_accumulator_);

  WriteMarker(out_stream, binary, "<NUM_OCCUPANCY>");
  num_occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DEN_OCCUPANCY>");
  den_occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIAGVARACCS>");
  variance_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</GMMEBWACCS>");
}


void AccumEbwDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
  num_occupancy_.Resize(num_comp);
  den_occupancy_.Resize(num_comp);
  if (flags_ & kGmmMeans)
    mean_accumulator_.Resize(num_comp, dim);
  else
    mean_accumulator_.Resize(0, 0);
  if (flags_ & kGmmVariances)
    variance_accumulator_.Resize(num_comp, dim);
  else
    variance_accumulator_.Resize(0, 0);
}

void AccumEbwDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) {
    num_occupancy_.SetZero();
    den_occupancy_.SetZero();
  }
  if (flags & kGmmMeans) mean_accumulator_.SetZero();
  if (flags & kGmmVariances) variance_accumulator_.SetZero();
}


void AccumEbwDiagGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) {
    num_occupancy_.Scale(d);
    den_occupancy_.SetZero();
  }
  if (flags & kGmmMeans) mean_accumulator_.Scale(d);
  if (flags & kGmmVariances) variance_accumulator_.Scale(d);
}

void AccumEbwDiagGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat>& data,
    const VectorBase<BaseFloat>& pos_post,
    const VectorBase<BaseFloat>& neg_post) {
  assert(static_cast<int32>(data.Dim()) == Dim());
  assert(static_cast<int32>(pos_post.Dim()) == NumGauss());
  Vector<double> pos_post_d(pos_post),
      neg_post_d(neg_post);  // Copy with type-conversion

  // accumulate
  num_occupancy_.AddVec(1.0, pos_post_d);
  num_occupancy_.AddVec(1.0, neg_post_d);
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    // TODO(arnab): we need to decide whether the neg posts have negative value
    mean_accumulator_.AddVecVec(1.0, pos_post_d, data_d);
    mean_accumulator_.AddVecVec(-1.0, neg_post_d, data_d);
    if (flags_ & kGmmVariances) {
      data_d.ApplyPow(2.0);
      variance_accumulator_.AddVecVec(1.0, pos_post_d, data_d);
      variance_accumulator_.AddVecVec(-1.0, neg_post_d, data_d);
    }
  }
}

void AccumEbwDiagGmm::SmoothWithAccum(BaseFloat tau, const AccumDiagGmm& src_acc) {
  KALDI_ASSERT(src_acc.NumGauss() == num_comp_ && src_acc.Dim() == dim_);
  double tau_d = static_cast<double>(tau);
  num_occupancy_.AddVec(tau_d, src_acc.occupancy());
  mean_accumulator_.AddMat(tau_d, src_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.AddMat(tau_d, src_acc.variance_accumulator(), kNoTrans);
}


void AccumEbwDiagGmm::SmoothWithModel(BaseFloat tau, const DiagGmm& gmm) {
  KALDI_ASSERT(gmm.NumGauss() == num_comp_ && gmm.Dim() == dim_);
  Matrix<double> means(num_comp_, dim_);
  Matrix<double> vars(num_comp_, dim_);
  gmm.GetMeans(&means);
  gmm.GetVars(&vars);

  mean_accumulator_.AddMat(tau, means);
  means.ApplyPow(2.0);
  vars.AddMat(1.0, means, kNoTrans);
  variance_accumulator_.AddMat(tau, vars);

  num_occupancy_.Add(tau);
}

AccumEbwDiagGmm::AccumEbwDiagGmm(const AccumEbwDiagGmm &other)
    : dim_(other.dim_), num_comp_(other.num_comp_),
      flags_(other.flags_), num_occupancy_(other.num_occupancy_),
      den_occupancy_(other.den_occupancy_),
      mean_accumulator_(other.mean_accumulator_),
      variance_accumulator_(other.variance_accumulator_) {}

}  // End of namespace kaldi
