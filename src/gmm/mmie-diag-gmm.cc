// gmm/mmie-diag-gmm.cc

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

#include <vector>

#include "gmm/diag-gmm.h"
#include "gmm/mmie-diag-gmm.h"

namespace kaldi {

void AccumDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectMarker(in_stream, binary, "<GMMACCS>");
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
        KALDI_ERR << "MlEstimatediagGmm::Read, dimension or flags mismatch, "
                  << (NumGauss()) + ", " << (Dim()) + ", "
                  << (Flags()) +" vs. " << (num_components) + ", "
                  << (dimension) + ", " << (flags);
      }
    } else {
      ResizeAccumulators(num_components, dimension, flags);
    }
  } else {
    ResizeAccumulators(num_components, dimension, flags);
  }

  ReadMarker(in_stream, binary, &token);
  while (token != "</GMMACCS>") {
    if (token == "<OCCUPANCY>") {
      occupancy_.Read(in_stream, binary, add);
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

void AccumDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<GMMACCS>");
  WriteMarker(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteMarker(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> occupancy_bf(occupancy_.Dim());
  Matrix<BaseFloat> mean_accumulator_bf(mean_accumulator_.NumRows(),
      mean_accumulator_.NumCols());
  Matrix<BaseFloat> variance_accumulator_bf(variance_accumulator_.NumRows(),
      variance_accumulator_.NumCols());
  occupancy_bf.CopyFromVec(occupancy_);
  mean_accumulator_bf.CopyFromMat(mean_accumulator_);
  variance_accumulator_bf.CopyFromMat(variance_accumulator_);

  WriteMarker(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIAGVARACCS>");
  variance_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</GMMACCS>");
}


void AccumDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentFlags(flags);
  occupancy_.Resize(num_comp);
  if (flags_ & kGmmMeans)
    mean_accumulator_.Resize(num_comp, dim);
  else
    mean_accumulator_.Resize(0, 0);
  if (flags_ & kGmmVariances)
    variance_accumulator_.Resize(num_comp, dim);
  else
    variance_accumulator_.Resize(0, 0);
}

void AccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) occupancy_.SetZero();
  if (flags & kGmmMeans) mean_accumulator_.SetZero();
  if (flags & kGmmVariances) variance_accumulator_.SetZero();
}


void AccumDiagGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) occupancy_.Scale(d);
  if (flags & kGmmMeans) mean_accumulator_.Scale(d);
  if (flags & kGmmVariances) variance_accumulator_.Scale(d);
}

void AccumDiagGmm::AccumulateForComponent(const VectorBase<BaseFloat>& data,
                                          int32 comp_index, BaseFloat weight) {
  assert(data.Dim() == Dim());
  double wt = static_cast<double>(weight);

  // accumulate
  occupancy_(comp_index) += wt;
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    mean_accumulator_.Row(comp_index).AddVec(wt, data_d);
    if (flags_ & kGmmVariances) {
      data_d.ApplyPow(2.0);
      variance_accumulator_.Row(comp_index).AddVec(wt, data_d);
    }
  }
}

void AccumDiagGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat>& data,
    const VectorBase<BaseFloat>& posteriors) {
  assert(static_cast<int32>(data.Dim()) == Dim());
  assert(static_cast<int32>(posteriors.Dim()) == NumGauss());
  Vector<double> post_d(posteriors);  // Copy with type-conversion

  // accumulate
  occupancy_.AddVec(1.0, post_d);
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    mean_accumulator_.AddVecVec(1.0, post_d, data_d);
    if (flags_ & kGmmVariances) {
      data_d.ApplyPow(2.0);
      variance_accumulator_.AddVecVec(1.0, post_d, data_d);
    }
  }
}

BaseFloat AccumDiagGmm::AccumulateFromDiag(const DiagGmm &gmm,
                                           const VectorBase<BaseFloat>& data,
                                           BaseFloat frame_posterior) {
  assert(gmm.NumGauss() == NumGauss());
  assert(gmm.Dim() == Dim());
  assert(static_cast<int32>(data.Dim()) == Dim());

  Vector<BaseFloat> posteriors(NumGauss());
  BaseFloat log_like = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

  AccumulateFromPosteriors(data, posteriors);
  return log_like;
}


void AccumDiagGmm::SmoothStats(BaseFloat tau) {
  Vector<double> smoothing_vec(occupancy_);
  smoothing_vec.InvertElements();
  smoothing_vec.Scale(static_cast<double>(tau));
  smoothing_vec.Add(1.0);

  mean_accumulator_.MulRowsVec(smoothing_vec);
  variance_accumulator_.MulRowsVec(smoothing_vec);
  occupancy_.Add(static_cast<double>(tau));
}


void AccumDiagGmm::SmoothWithAccum(BaseFloat tau, const AccumDiagGmm& src_acc) {
  KALDI_ASSERT(src_acc.NumGauss() == num_comp_ && src_acc.Dim() == dim_);
  double tau_d = static_cast<double>(tau);
  occupancy_.AddVec(tau_d, src_acc.occupancy_);
  mean_accumulator_.AddMat(tau_d, src_acc.mean_accumulator_, kNoTrans);
  variance_accumulator_.AddMat(tau_d, src_acc.variance_accumulator_, kNoTrans);
}


void AccumDiagGmm::SmoothWithModel(BaseFloat tau, const DiagGmm& gmm) {
  KALDI_ASSERT(gmm.NumGauss() == num_comp_ && gmm.Dim() == dim_);
  // this won't compile now because of the precision issues... wait for the DiagGmmNormal class to be implemented
  Matrix<double> means(num_comp_, dim_);
  Matrix<double> vars(num_comp_, dim_);
  gmm.GetMeans(&means);
  gmm.GetVars(&vars);

  mean_accumulator_.AddMat(tau, means);
  means.ApplyPow(2.0);
  vars.AddMat(1.0, means, kNoTrans);
  variance_accumulator_.AddMat(tau, vars);

  occupancy_.Add(tau);
}

BaseFloat MmieDiagGmm::MmiObjective(const DiagGmm& gmm) const {
}

}  // End of namespace kaldi
