// gmm/mle-full-gmm.cc

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation;  Georg Stemmer
//                      Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include <string>

#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"

namespace kaldi {


AccumFullGmm::AccumFullGmm(const AccumFullGmm &other)
  : dim_(other.dim_), num_comp_(other.num_comp_),
    flags_(other.flags_), occupancy_(other.occupancy_),
    mean_accumulator_(other.mean_accumulator_),
    covariance_accumulator_(other.covariance_accumulator_) {}

void AccumFullGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
  occupancy_.Resize(num_comp);
  if (flags_ & kGmmMeans)
    mean_accumulator_.Resize(num_comp, dim);
  else
    mean_accumulator_.Resize(0, 0);
    
  if (flags_ & kGmmVariances)
    ResizeVarAccumulator(num_comp, dim);
  else
    covariance_accumulator_.clear();
}

void AccumFullGmm::ResizeVarAccumulator(int32 num_comp, int32 dim) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  if (covariance_accumulator_.size() != static_cast<size_t>(num_comp))
    covariance_accumulator_.resize(num_comp);
  for (int32 i = 0; i < num_comp; i++) {
    if (covariance_accumulator_[i].NumRows() != dim)
      covariance_accumulator_[i].Resize(dim);
  }
}

void AccumFullGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  
  if (flags & kGmmWeights) 
    occupancy_.SetZero();
  
  if (flags & kGmmMeans) 
    mean_accumulator_.SetZero();
  
  if (flags & kGmmVariances) {
    for (int32 i = 0, end = covariance_accumulator_.size(); i < end; i++)
      covariance_accumulator_[i].SetZero();
  }
}

void AccumFullGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
    
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) 
    occupancy_.Scale(d);
  
  if (flags & kGmmMeans) 
    mean_accumulator_.Scale(d);
  
  if (flags & kGmmVariances) {
    for (int32 i = 0, end = covariance_accumulator_.size(); i < end; i++)
      covariance_accumulator_[i].Scale(d);
  }
}

void AccumFullGmm::AccumulateForComponent(
    const VectorBase<BaseFloat> &data, int32 comp_index, BaseFloat weight) {
  KALDI_ASSERT(data.Dim() == Dim());
  double wt = static_cast<double>(weight);

  // accumulate
  occupancy_(comp_index) += wt;
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    mean_accumulator_.Row(comp_index).AddVec(wt, data_d);
    if (flags_ & kGmmVariances) {
      covariance_accumulator_[comp_index].AddVec2(wt, data_d);
    }
  }
}

void AccumFullGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &gauss_posteriors) {
  KALDI_ASSERT(gauss_posteriors.Dim() == NumGauss());
  KALDI_ASSERT(data.Dim() == Dim());
  Vector<double> data_d(data.Dim());
  data_d.CopyFromVec(data);
  Vector<double> post_d(gauss_posteriors.Dim());
  post_d.CopyFromVec(gauss_posteriors);

  occupancy_.AddVec(1.0, post_d);
  if (flags_ & (kGmmMeans|kGmmVariances)) {  // mean stats.
    if (static_cast<int32>(post_d.Norm(0.0)*2.0) > post_d.Dim()) {
      // If we're not very sparse... note: zero-norm is number of
      // nonzero elements.
      mean_accumulator_.AddVecVec(1.0, post_d, data_d);
    } else {
      for (int32 i = 0; i < post_d.Dim(); i++)
        if (post_d(i) != 0.0)
          mean_accumulator_.Row(i).AddVec(post_d(i), data_d);
    }
    if (flags_ & kGmmVariances) {
      SpMatrix<double> data_sq_d(data_d.Dim());
      data_sq_d.AddVec2(1.0, data_d);
      for (int32 mix = 0; mix < NumGauss(); mix++)
        if (post_d(mix) !=  0.0)
          covariance_accumulator_[mix].AddSp(post_d(mix), data_sq_d);
    }
  }
}

BaseFloat AccumFullGmm::AccumulateFromFull(const FullGmm &gmm,
    const VectorBase<BaseFloat> &data, BaseFloat frame_posterior) {
  KALDI_ASSERT(gmm.NumGauss() == NumGauss());
  KALDI_ASSERT(gmm.Dim() == Dim());

  Vector<BaseFloat> component_posterior(NumGauss());

  BaseFloat log_like = gmm.ComponentPosteriors(data, &component_posterior);
  component_posterior.Scale(frame_posterior);

  AccumulateFromPosteriors(data, component_posterior);
  return log_like;
}

BaseFloat AccumFullGmm::AccumulateFromDiag(const DiagGmm &gmm,
    const VectorBase<BaseFloat> &data, BaseFloat frame_posterior) {
  KALDI_ASSERT(gmm.NumGauss() == NumGauss());
  KALDI_ASSERT(gmm.Dim() == Dim());

  Vector<BaseFloat> component_posterior(NumGauss());

  BaseFloat log_like = gmm.ComponentPosteriors(data, &component_posterior);
  component_posterior.Scale(frame_posterior);

  AccumulateFromPosteriors(data, component_posterior);
  return log_like;
}

void AccumFullGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectToken(in_stream, binary, "<GMMACCS>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectToken(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  KALDI_ASSERT(dimension > 0 && num_components > 0);
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

  // these are needed for demangling the variances.
  Vector<double> tmp_occs;
  Matrix<double> tmp_means;

  ReadToken(in_stream, binary, &token);
  while (token != "</GMMACCS>") {
    if (token == "<OCCUPANCY>") {
      tmp_occs.Read(in_stream, binary, false);
      if (!add) occupancy_.SetZero();
      occupancy_.AddVec(1.0, tmp_occs);
    } else if (token == "<MEANACCS>") {
      tmp_means.Read(in_stream, binary, false);
      if (!add) mean_accumulator_.SetZero();
      mean_accumulator_.AddMat(1.0, tmp_means);
    } else if (token == "<FULLVARACCS>") {
      for (int32 i = 0; i < num_components; i++) {
        SpMatrix<double> tmp_acc;
        tmp_acc.Read(in_stream, binary, add);
        if (tmp_occs(i) != 0) tmp_acc.AddVec2(1.0 / tmp_occs(i), tmp_means.Row(
            i));
        if (!add) covariance_accumulator_[i].SetZero();
        covariance_accumulator_[i].AddSp(1.0, tmp_acc);
      }
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadToken(in_stream, binary, &token);
  }
}

void AccumFullGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteToken(out_stream, binary, "<GMMACCS>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteToken(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteToken(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  Vector<BaseFloat> occupancy_bf(occupancy_);
  WriteToken(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  Matrix<BaseFloat> mean_accumulator_bf(mean_accumulator_);
  WriteToken(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);

  if (num_comp_ != 0) KALDI_ASSERT(((flags_ & kGmmVariances) != 0 )
      == (covariance_accumulator_.size() != 0));  // sanity check.
  if (covariance_accumulator_.size() != 0) {
    WriteToken(out_stream, binary, "<FULLVARACCS>");
    for (int32 i = 0; i < num_comp_; i++) {
      SpMatrix<double> tmp_acc(covariance_accumulator_[i]);
      if (occupancy_(i) != 0) tmp_acc.AddVec2(-1.0 / occupancy_(i),
          mean_accumulator_.Row(i));
      SpMatrix<float> tmp_acc_bf(tmp_acc);
      tmp_acc_bf.Write(out_stream, binary);
    }
  }
  WriteToken(out_stream, binary, "</GMMACCS>");
}

BaseFloat MlObjective(const FullGmm &gmm, const AccumFullGmm &fullgmm_acc) {
  GmmFlagsType flags = fullgmm_acc.Flags();
  Vector<BaseFloat> occ_bf(fullgmm_acc.occupancy());
  Matrix<BaseFloat> mean_accs_bf(fullgmm_acc.mean_accumulator());
  SpMatrix<BaseFloat> covar_accs_bf(gmm.Dim());
 
  BaseFloat obj = VecVec(occ_bf, gmm.gconsts());
  
  if (flags & kGmmMeans)
    obj += TraceMatMat(mean_accs_bf, gmm.means_invcovars(), kTrans);
  
  if (flags & kGmmVariances) {
    for (int32 i = 0; i < gmm.NumGauss(); i++) {
      covar_accs_bf.CopyFromSp(fullgmm_acc.covariance_accumulator()[i]);
      obj -= 0.5 * TraceSpSp(covar_accs_bf, gmm.inv_covars()[i]);
    }
  }
  
  return obj;
}

void MleFullGmmUpdate(const MleFullGmmOptions &config,
                      const AccumFullGmm &fullgmm_acc,
                      GmmFlagsType flags,
                      FullGmm *gmm,
                      BaseFloat *obj_change_out,
                      BaseFloat *count_out) {
  KALDI_ASSERT(gmm != NULL);
  
  if (flags & ~fullgmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, fullgmm_acc);

  // Korbinian: I removed checks that validate if the referenced gmm matches
  // the accumulator, as this should be responsibility of the caller.
  // Furthermore, the re-estimation of the normal representation is done 
  // regardless of the flags, but the transfer to the natural form is
  // done with respect to the flags.

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = fullgmm_acc.occupancy().Sum();

  int32 tot_floored = 0, gauss_floored = 0;

  // allocate the gmm in normal representation
  FullGmmNormal ngmm(*gmm);

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; i++) {
    double occ = fullgmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_gauss;
    
    if (occ > static_cast<double> (config.min_gaussian_occupancy)
        && prob > static_cast<double> (config.min_gaussian_weight)) {
      
      ngmm.weights_(i) = prob;

      // copy old mean for later normalizations
      Vector<double> oldmean(ngmm.means_.Row(i));

      // update mean, then variance, as far as there are accumulators
      if (fullgmm_acc.Flags() & (kGmmMeans|kGmmVariances)) {
        Vector<double> mean(fullgmm_acc.mean_accumulator().Row(i));
        mean.Scale(1.0 / occ);

        // transfer to estimate
        ngmm.means_.CopyRowFromVec(mean, i);
      }

      if (fullgmm_acc.Flags() & kGmmVariances) {
        KALDI_ASSERT(fullgmm_acc.Flags() & kGmmMeans);
        SpMatrix<double> covar(fullgmm_acc.covariance_accumulator()[i]);
        covar.Scale(1.0 / occ);
        covar.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
        // if we intend to only update the variances, we need to compensate by 
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          oldmean.AddVec(-1.0, ngmm.means_.Row(i));
          covar.AddVec2(1.0, oldmean);
        }

        // Now flooring etc. of variance's eigenvalues.
        BaseFloat floor = std::max(static_cast<double>(config.variance_floor),
                                   covar.MaxAbsEig() / config.max_condition);
        
        int32 floored = covar.ApplyFloor(floor);

        if (floored) {
          tot_floored += floored;
          gauss_floored++;
        }

        // transfer to estimate
        ngmm.vars_[i].CopyFromSp(covar);
      }
    } else {  // Insufficient occupancy
      if (config.remove_low_count_gaussians &&
            static_cast<int32>(to_remove.size()) < num_gauss-1) {
        KALDI_WARN << "Too little data - removing Gaussian (weight "
                   << std::fixed << prob
                   << ", occupation count " << std::fixed << fullgmm_acc.occupancy()(i)
                   << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
                   << (config.remove_low_count_gaussians ?
                       " it is the last Gaussian: i = "
                       : " remove-low-count-gaussians == false: i = ") << i
                   << ", occ = " << fullgmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }

  // copy to natural representation according to flags
  ngmm.CopyToFullGmm(gmm, flags);

  gmm->ComputeGconsts();
  BaseFloat obj_new = MlObjective(*gmm, fullgmm_acc);
  
  if (obj_change_out)
    *obj_change_out = obj_new - obj_old;
  
  if (count_out) 
    *count_out = occ_sum;

  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /* renorm weights */);
    gmm->ComputeGconsts();
  }
  
  if (tot_floored > 0)
    KALDI_WARN << tot_floored << " variances floored in " << gauss_floored
               << " Gaussians.";

}

}  // End namespace kaldi
