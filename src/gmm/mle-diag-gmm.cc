// gmm/mle-diag-gmm.cc

// Copyright 2009-2011  Saarland University;  Georg Stemmer;  Jan Silovsky;
//                      Microsoft Corporation; Yanmin Qian

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
#include "gmm/mle-diag-gmm.h"

namespace kaldi {

void AccumDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectToken(in_stream, binary, "<GMMACCS>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectToken(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  ExpectToken(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if ((NumGauss() != 0 || Dim() != 0 || Flags() != 0)) {
      if (num_components != NumGauss() || dimension != Dim()
          || flags != Flags())
        KALDI_ERR << "MlEstimatediagGmm::Read, dimension or flags mismatch, "
                  << NumGauss() << ", " << Dim() << ", "
                  << GmmFlagsToString(Flags()) << " vs. " << num_components << ", "
                  << dimension << ", " << flags;
    } else {
      Resize(num_components, dimension, flags);
    }
  } else {
    Resize(num_components, dimension, flags);
  }

  ReadToken(in_stream, binary, &token);
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
    ReadToken(in_stream, binary, &token);
  }
}

void AccumDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteToken(out_stream, binary, "<GMMACCS>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteToken(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteToken(out_stream, binary, "<FLAGS>");
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

  WriteToken(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<DIAGVARACCS>");
  variance_accumulator_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "</GMMACCS>");
}


void AccumDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
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
  if (flags_ & kGmmMeans)
    KALDI_ASSERT(data.Dim() == Dim());
  double wt = static_cast<double>(weight);
  KALDI_ASSERT(comp_index < NumGauss());
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

void AccumDiagGmm::AddStatsForComponent(int32 g,
                                        double occ,
                                        const VectorBase<double> &x_stats,
                                        const VectorBase<double> &x2_stats) {
  KALDI_ASSERT(g < NumGauss());
  occupancy_(g) += occ;
  if (flags_ & kGmmMeans)
    mean_accumulator_.Row(g).AddVec(1.0, x_stats);
  if (flags_ & kGmmVariances)
    variance_accumulator_.Row(g).AddVec(1.0, x2_stats);
}


void AccumDiagGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat>& data,
    const VectorBase<BaseFloat>& posteriors) {
  if (flags_ & kGmmMeans)
    KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());
  KALDI_ASSERT(static_cast<int32>(posteriors.Dim()) == NumGauss());
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
  KALDI_ASSERT(gmm.NumGauss() == NumGauss());
  KALDI_ASSERT(gmm.Dim() == Dim());
  KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());

  Vector<BaseFloat> posteriors(NumGauss());
  BaseFloat log_like = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

  AccumulateFromPosteriors(data, posteriors);
  return log_like;
}


// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothStats(BaseFloat tau) {
  Vector<double> smoothing_vec(occupancy_);
  smoothing_vec.InvertElements();
  smoothing_vec.Scale(static_cast<double>(tau));
  smoothing_vec.Add(1.0);
  // now smoothing_vec = (tau + occ) / occ

  mean_accumulator_.MulRowsVec(smoothing_vec);
  variance_accumulator_.MulRowsVec(smoothing_vec);
  occupancy_.Add(static_cast<double>(tau));
}


// want to add tau "virtual counts" of each Gaussian from "src_acc"
// to each Gaussian in this acc.
// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothWithAccum(BaseFloat tau, const AccumDiagGmm& src_acc) {
  KALDI_ASSERT(src_acc.NumGauss() == num_comp_ && src_acc.Dim() == dim_);
  for (int32 i = 0; i < num_comp_; i++) {
    if (src_acc.occupancy_(i) != 0.0) { // can only smooth if src was nonzero...
      occupancy_(i) += tau;
      mean_accumulator_.Row(i).AddVec(tau / src_acc.occupancy_(i),
                                      src_acc.mean_accumulator_.Row(i));
      variance_accumulator_.Row(i).AddVec(tau / src_acc.occupancy_(i),
                                          src_acc.variance_accumulator_.Row(i));
    } else
      KALDI_WARN << "Could not smooth since source acc had zero occupancy.";
  }
}


void AccumDiagGmm::SmoothWithModel(BaseFloat tau, const DiagGmm& gmm) {
  KALDI_ASSERT(gmm.NumGauss() == num_comp_ && gmm.Dim() == dim_);
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

AccumDiagGmm::AccumDiagGmm(const AccumDiagGmm &other)
    : dim_(other.dim_), num_comp_(other.num_comp_),
      flags_(other.flags_), occupancy_(other.occupancy_),
      mean_accumulator_(other.mean_accumulator_),
      variance_accumulator_(other.variance_accumulator_) {}

int32 FloorVariance(const  VectorBase<BaseFloat> &variance_floor_vector,
                           VectorBase<double> *var) {
  int32 ans = 0;
  KALDI_ASSERT(variance_floor_vector.Dim() == var->Dim());
  for (int32 i = 0; i < var->Dim(); i++) {
    if ((*var)(i) < variance_floor_vector(i)) {
      (*var)(i) = variance_floor_vector(i);
      ans++;
    }
  }
  return ans;
}

int32 FloorVariance(const BaseFloat min_variance,
                          VectorBase<double> *var) {
  int32 ans = 0;
  for (int32 i = 0; i < var->Dim(); i++) {
    if ((*var)(i) < min_variance) {
      (*var)(i) = min_variance;
      ans++;
    }
  }
  return ans;
}

BaseFloat MlObjective(const DiagGmm& gmm,
                      const AccumDiagGmm &diaggmm_acc) {
  GmmFlagsType acc_flags = diaggmm_acc.Flags();
  Vector<BaseFloat> occ_bf(diaggmm_acc.occupancy());
  Matrix<BaseFloat> mean_accs_bf(diaggmm_acc.mean_accumulator());
  Matrix<BaseFloat> variance_accs_bf(diaggmm_acc.variance_accumulator());
  BaseFloat obj = VecVec(occ_bf, gmm.gconsts());
  if (acc_flags & kGmmMeans)
    obj += TraceMatMat(mean_accs_bf, gmm.means_invvars(), kTrans);
  if (acc_flags & kGmmVariances)
    obj -= 0.5 * TraceMatMat(variance_accs_bf, gmm.inv_vars(), kTrans);
  return obj;
}

void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diaggmm_acc,
                      GmmFlagsType flags,
                      DiagGmm *gmm,
                      BaseFloat *obj_change_out,
                      BaseFloat *count_out) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diaggmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  
  // Korbinian: I removed checks that validate if the referenced gmm matches
  // the accumulator, as this should be responsibility of the caller.
  // Furthermore, the re-estimation of the normal representation is done 
  // regardless of the flags, but the transfer to the natural form is
  // done with respect to the flags.

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diaggmm_acc.occupancy().Sum();

  int32 tot_floored = 0, gauss_floored = 0;

  // remember old objective value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diaggmm_acc);

  // allocate the gmm in normal representation; all parameters of this will be 
  // updated, but only the flagged ones will be transferred back to gmm
  DiagGmmNormal ngmm(*gmm);

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; ++i) {
    double occ = diaggmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_gauss;

    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
      
      ngmm.weights_(i) = prob;
      
      // copy old mean for later normalizations
      Vector<double> oldmean(ngmm.means_.Row(i));
      
      // update mean, then variance, as far as there are accumulators 
      if (diaggmm_acc.Flags() & kGmmMeans) {
        Vector<double> mean(diaggmm_acc.mean_accumulator().Row(i));
        mean.Scale(1.0 / occ);

        // transfer to estimate
        ngmm.means_.CopyRowFromVec(mean, i);
      }
      
      if (diaggmm_acc.Flags() & kGmmVariances) {
        KALDI_ASSERT(diaggmm_acc.Flags() & kGmmMeans);
        Vector<double> var(diaggmm_acc.variance_accumulator().Row(i));
        var.Scale(1.0 / occ);
        var.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
     
        // if we intend to only update the variances, we need to compensate by 
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          oldmean.AddVec(-1.0, ngmm.means_.Row(i));
          var.AddVec2(1.0, oldmean);
        }
   
        int32 floored;
        if (config.variance_floor_vector.Dim() != 0)
          floored = FloorVariance(config.variance_floor_vector, &var);
        else 
          floored = FloorVariance(config.min_variance, &var);
      
        if (floored) {
          tot_floored += floored;
          gauss_floored++;
        }

        // transfer to estimate
        ngmm.vars_.CopyRowFromVec(var, i);
      }
      
    } else {  // Insufficient occupancy.
      if (config.remove_low_count_gaussians &&
          static_cast<int32>(to_remove.size()) < num_gauss-1) {
        // remove the component, unless it is the last one.
        KALDI_WARN << "Too little data - removing Gaussian (weight "
                   << std::fixed << prob
                   << ", occupation count " << std::fixed << diaggmm_acc.occupancy()(i)
                   << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
                   << (config.remove_low_count_gaussians ?
                       " it is the last Gaussian: i = "
                       : " remove-low-count-gaussians == false: g = ") << i
                   << ", occ = " << diaggmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }
  
  // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diaggmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  
  if (count_out) *count_out = occ_sum;
  
  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights*/);
    gmm->ComputeGconsts();
  }

  if (tot_floored > 0)
    KALDI_WARN << tot_floored << " variances floored in " << gauss_floored
               << " Gaussians.";
}

void AccumDiagGmm::Add(double scale, const AccumDiagGmm &acc) {
  // The functions called here will crash if the dimensions etc.
  // or the flags don't match.
  occupancy_.AddVec(scale, acc.occupancy_);
  if (flags_ & kGmmMeans)
    mean_accumulator_.AddMat(scale, acc.mean_accumulator_);
  if (flags_ & kGmmVariances)
    variance_accumulator_.AddMat(scale, acc.variance_accumulator_);
}



}  // End of namespace kaldi
