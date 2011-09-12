// tied/mle-tied-gmm.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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
#include <sstream>

#include "tied/tied-gmm.h"
#include "tied/mle-tied-gmm.h"

namespace kaldi {

void AccumTiedGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 num_components;
  std::string token;
  GmmFlagsType flags;

  ExpectMarker(in_stream, binary, "<TIEDGMMACCS>");
  ExpectMarker(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  ExpectMarker(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if (NumGauss() != 0) {
      if (num_components != NumGauss() || Flags() != 0) {
        KALDI_ERR << "AccumTiedDiagGmm::Read, num_components mismatch "
                  << (NumGauss()) << ", " << (Flags()) << " vs "
                  << (num_components) << ", " << (flags);
      }
    } else {
      Resize(num_components, flags);
    }
  } else {
    Resize(num_components, flags);
  }

  ReadMarker(in_stream, binary, &token);
  while (token != "</TIEDGMMACCS>") {
    if (token == "<OCCUPANCY>") {
      occupancy_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void AccumTiedGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<TIEDGMMACCS>");
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteMarker(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> occupancy_bf(occupancy_.Dim());
  occupancy_bf.CopyFromVec(occupancy_);

  WriteMarker(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</TIEDGMMACCS>");
}

void AccumTiedGmm::Resize(int32 num_comp, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0);
  num_comp_ = num_comp;
  flags_ = AugmentGmmFlags(flags);
  occupancy_.Resize(num_comp);
}

void AccumTiedGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) occupancy_.SetZero();
}

void AccumTiedGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) occupancy_.Scale(d);
}

void AccumTiedGmm::AccumulateForComponent(int32 comp_index, BaseFloat weight) {
  assert(comp_index < static_cast<int32>(occupancy_.Dim()));
  occupancy_(comp_index) += static_cast<double>(weight);
}

void AccumTiedGmm::AccumulateFromPosteriors(
       const VectorBase<BaseFloat> &posteriors) {
  assert(static_cast<int32>(posteriors.Dim()) == NumGauss());
  Vector<double> post_d(posteriors);  // Copy with type-conversion
  occupancy_.AddVec(1.0, post_d);
}

/// Propagate the sufficient statistics to the target accumulator
void AccumTiedGmm::Propagate(AccumTiedGmm *target) const {
  KALDI_ASSERT(num_comp_ == target->num_comp_);
  target->occupancy_.AddVec(1.0, occupancy_);
}

/// Interpolate the local model depending on the occupancies
/// rho' <- rho / (rho + gamma)
/// this <- rho' x source + (1-rho') x this
void AccumTiedGmm::Interpolate(BaseFloat rho, const AccumTiedGmm *source) {
  KALDI_ASSERT(num_comp_ == source->num_comp_);
  BaseFloat rhoi = rho / (rho + occupancy_.Sum());

  if (rhoi > 0.8)
    KALDI_VLOG(1) << "rhoi > 0.8";

  occupancy_.Scale(1.0 - rhoi);
  occupancy_.AddVec(rhoi, source->occupancy_);
}

AccumTiedGmm::AccumTiedGmm(const AccumTiedGmm &other)
    : num_comp_(other.num_comp_),
      flags_(other.flags_), occupancy_(other.occupancy_)
      {}

BaseFloat MlObjective(const TiedGmm &tied, const AccumTiedGmm &tiedgmm_acc) {
  // use the occupancy of the tied pdf
  Vector<BaseFloat> occ_bf(tiedgmm_acc.occupancy());
  Vector<BaseFloat> logwt(tied.weights());
  logwt.ApplyLog();
  return VecVec(occ_bf, logwt);
}

void MleTiedGmmUpdate(const MleTiedGmmOptions &config,
            const AccumTiedGmm &tiedgmm_acc,
            GmmFlagsType flags,
            TiedGmm *tied,
            BaseFloat *obj_change_out,
            BaseFloat *count_out) {
  if (flags & ~tiedgmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  if (!(flags & kGmmWeights)) {
    KALDI_WARN << "no weight update as desired by flags -- why would you call "
                  "MleTiedGmmUpdate in the first place?";
    return;
  }

  double occ_sum = tiedgmm_acc.occupancy().Sum();
  int32 num_comp = tiedgmm_acc.occupancy().Dim();

  KALDI_ASSERT(tied->NumGauss() == num_comp);

  if (occ_sum <= config.min_gaussian_occupancy) {
    KALDI_WARN << "Total occupancy of this TiedGmm too small, skipping weight "
                  "update!";
  } else {
    // floor the weights with respect to the number of components
    double floor = config.min_gaussian_weight / num_comp;

    // remember old objective value
    BaseFloat obj_old = MlObjective(*tied, tiedgmm_acc);

    // update weights
    std::vector<int32> floored_weights;
    for (int32 g = 0; g < num_comp; g++) {
      double wt = tiedgmm_acc.occupancy()(g) / occ_sum;

      if (wt < floor) {
        // KALDI_WARN << "Weight of component " << g << " too small
        // (" << wt << "), flooring to min_gaussian_weight";
        wt = floor;
        floored_weights.push_back(g);
      }

      tied->SetComponentWeight(g, wt);
    }

    if (floored_weights.size() > 0) {
      // to correct for the min weights
      // we need to re-normalize the weights
      Vector<BaseFloat> w(tied->weights());
      w.Scale(1.0 / w.Sum());
      tied->SetWeights(w);

      KALDI_WARN << "Floored " << floored_weights.size() << " weights to "
                 << floor;
    }

    // compute new objective value
    BaseFloat obj_new = MlObjective(*tied, tiedgmm_acc);

    if (obj_change_out)
      *obj_change_out = (obj_new - obj_old);

    if (count_out)
      *count_out = occ_sum;
  }
}

}  // End of namespace kaldi
