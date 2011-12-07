// gmm/mmie-diag-gmm.cc

// Copyright 2009-2011 Petr Motlicek, Arnab Ghoshal

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
#include "gmm/mmie-diag-gmm.h"


namespace kaldi {


void MmieAccumDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectMarker(in_stream, binary, "<GMMMMIACCS>");
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
  while (token != "</GMMMMIACCS>") {
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
  /// get difference
  occupancy_.CopyFromVec(num_occupancy_);
  occupancy_.AddVec(-1.0, den_occupancy_);

}

void MmieAccumDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<GMMMMIACCS>");
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
  WriteMarker(out_stream, binary, "</GMMMMIACCS>");
}




void MmieAccumDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
  num_occupancy_.Resize(num_comp);
  den_occupancy_.Resize(num_comp);
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


void MmieAccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) {
    num_occupancy_.SetZero();
    den_occupancy_.SetZero();
    occupancy_.SetZero();
  }
  if (flags & kGmmMeans) mean_accumulator_.SetZero();
  if (flags & kGmmVariances) variance_accumulator_.SetZero();
}


void MmieAccumDiagGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) {
    num_occupancy_.Scale(d);
    den_occupancy_.Scale(d);
    occupancy_.Scale(d);
  }
  if (flags & kGmmMeans) mean_accumulator_.Scale(d);
  if (flags & kGmmVariances) variance_accumulator_.Scale(d);
}


void MmieAccumDiagGmm::SubtractAccumulatorsISmoothing(
    const AccumDiagGmm& num_acc,
    const AccumDiagGmm& den_acc,
    const MmieDiagGmmOptions& opts,
    const AccumDiagGmm& i_smooth_acc){
  
  //KALDI_ASSERT(num_acc.NumGauss() == den_acc.NumGauss && num_acc.Dim() == den_acc.Dim());
  //std::cout << "NumGauss: " << num_acc.NumGauss() << " " << den_acc.NumGauss() << " " << num_comp_ << '\n';
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_ && den_acc.Dim() == dim_);
  KALDI_ASSERT(i_smooth_acc.NumGauss() == num_comp_ && i_smooth_acc.Dim() == dim_);
  
  // no subracting occs, just copy them to local vars
  num_occupancy_.CopyFromVec(num_acc.occupancy());
  den_occupancy_.CopyFromVec(den_acc.occupancy());
  occupancy_.CopyFromVec(num_occupancy_);
  occupancy_.AddVec(-1.0, den_occupancy_);

  // Copy nums to private vars
  mean_accumulator_.CopyFromMat(num_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.CopyFromMat(num_acc.variance_accumulator(), kNoTrans);

  // Copy I- smoothing stats
  Vector<double> i_smooth_occupancy(i_smooth_acc.occupancy()); 
  Matrix<double> i_smooth_mean_accumulator(i_smooth_acc.mean_accumulator());
  Matrix<double> i_smooth_variance_accumulator(i_smooth_acc.variance_accumulator());
  // I- smoothing
  for (int32 g = 0; g < num_comp_; g++) {
    double occ = i_smooth_occupancy(g);
    if (occ >= 0.0) {
      occupancy_(g) += opts.i_smooth_tau; // Add I-smoothing to occupancy_, but
      // *not* to num_occupancy_, which remains the original count before
      // I-smoothing, and which we use to update the weights.
      mean_accumulator_.Row(g).AddVec(opts.i_smooth_tau/occ,
                                      i_smooth_mean_accumulator.Row(g));
      variance_accumulator_.Row(g).AddVec(opts.i_smooth_tau/occ,
                                      i_smooth_variance_accumulator.Row(g));
    }
  }  
  // Subtract den from smoothed num
  mean_accumulator_.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.AddMat(-1.0, den_acc.variance_accumulator(), kNoTrans);  
}


bool MmieAccumDiagGmm::EBWUpdateGaussian(
    BaseFloat D,
    GmmFlagsType flags,
    const VectorBase<double> &orig_mean,
    const VectorBase<double> &orig_var,
    const VectorBase<double> &x_stats,
    const VectorBase<double> &x2_stats,
    double occ,
    VectorBase<double> *mean,
    VectorBase<double> *var,
    double *auxf_impr) const {
  if (! (flags&(kGmmMeans|kGmmVariances)) || occ <= 0.0) { // nothing to do.
    if (auxf_impr) *auxf_impr = 0.0;
    mean->CopyFromVec(orig_mean);
    var->CopyFromVec(orig_var);
    return true; 
  }    
  KALDI_ASSERT(!( (flags&kGmmVariances) && !(flags&kGmmMeans)));
  
  mean->SetZero();
  var->SetZero();
  mean->AddVec(D, orig_mean);
  var->AddVec2(D, orig_mean);
  var->AddVec(D, orig_var);
  mean->AddVec(1.0, x_stats);
  var->AddVec(1.0, x2_stats);
  BaseFloat scale = 1.0 / (occ + D);
  mean->Scale(scale);
  var->Scale(scale);
  var->AddVec2(-1.0, *mean);

  if (!(flags&kGmmVariances)) var->CopyFromVec(orig_var);
  if (!(flags&kGmmMeans)) mean->CopyFromVec(orig_mean);
  
  if (var->Min() > 0.0) {
    if (auxf_impr != NULL) {
      // work out auxf improvement.  
      BaseFloat old_auxf = 0.0, new_auxf = 0.0;
      int32 dim = orig_mean.Dim();
      for (int32 i = 0; i < dim; i++) {
        BaseFloat mean_diff = (*mean)(i) - orig_mean(i);
        old_auxf += (occ+D) * -0.5 * (log(orig_var(i)) +
                                      ((*var)(i) + mean_diff*mean_diff)
                                      / orig_var(i));
        new_auxf += (occ+D) * -0.5 * (log((*var)(i)) + 1.0);
        
      }
      *auxf_impr = new_auxf - old_auxf;
    }
    return true;
  } else return false;
}


void MmieAccumDiagGmm::Update(const MmieDiagGmmOptions &config,
                              GmmFlagsType flags,
                              DiagGmm *gmm,
                              BaseFloat *auxf_change_out_gauss,
                              BaseFloat *auxf_change_out_weights,
                              BaseFloat *count_out,
                              int32 *num_floored_out) const {
  if (flags_ & ~flags)
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  if (auxf_change_out_gauss) *auxf_change_out_gauss = 0.0;
  if (auxf_change_out_weights) *auxf_change_out_weights = 0.0;
  if (count_out) *count_out = 0.0;
  if (num_floored_out) *num_floored_out = 0;
  
  KALDI_ASSERT(gmm->NumGauss() == (num_comp_));
  if (flags_ & kGmmMeans)
    KALDI_ASSERT(dim_ == mean_accumulator_.NumCols());
  
  int32 num_comp = num_comp_;
  int32 dim = dim_;
  
  // copy DiagGMM model and transform this to the normal case
  DiagGmmNormal diaggmmnormal;
  gmm->ComputeGconsts();
  diaggmmnormal.CopyFromDiagGmm(*gmm);
  
  // go over all components
  double occ;
  Vector<double> mean(dim), var(dim);
  for (int32 g = 0; g < num_comp; g++) {
    double D = config.ebw_e * den_occupancy_(g) / 2; // E*y_den/2 where E = 2;
    // We initialize to half the value of D that would be dicated by
    // E; this is part of the strategy used to ensure that the value of
    // D we use is at least twice the value that would ensure positive
    // variances.

    occ = occupancy_(g);

    int32 iter, max_iter = 100;
    for (iter = 0; iter < max_iter; iter++) { // will normally break the first time.
      if (EBWUpdateGaussian(D, flags,
                            diaggmmnormal.means_.Row(g),
                            diaggmmnormal.vars_.Row(g),
                            mean_accumulator_.Row(g),
                            variance_accumulator_.Row(g),
                            occ,
                            &mean,
                            &var,
                            NULL)) {
        // Succeeded in getting all +ve vars at this value of D.
        // So double D and commit changes.
        D *= 2.0;
        double auxf_impr = 0.0;
        EBWUpdateGaussian(D, flags,
                          diaggmmnormal.means_.Row(g),
                          diaggmmnormal.vars_.Row(g),                             
                          mean_accumulator_.Row(g),
                          variance_accumulator_.Row(g),
                          occ,
                          &mean,
                          &var,
                          &auxf_impr);
        if (auxf_change_out_gauss) *auxf_change_out_gauss += auxf_impr;
        if (count_out) *count_out += num_occupancy_(g);
        // the EBWUpdateGaussian function only updates the
        // appropriate parameters according to the flags.
        // variance flooring
        //for (int32 i = 0; i < var.Dim(); i++) {
        //  if (var(i) < config.min_variance) {
        //    var(i) = config.min_variance;
        //    KALDI_WARN << " flooring variance with value = " << var(i); 
        //  }
        //}
        diaggmmnormal.means_.CopyRowFromVec(mean, g);
        diaggmmnormal.vars_.CopyRowFromVec(var, g);
        
        break;
      } else {
        // small step
        D *= 1.1; 
      }
    }
    if (iter > 0 && num_floored_out != NULL) *num_floored_out++;
    if (iter == max_iter) KALDI_WARN << "Dropped off end of loop, recomputing D. (unexpected.)";
  }

  // Now update weights...
  if (flags & kGmmWeights) {
    double weight_auxf_at_start = 0.0, weight_auxf_at_end = 0.0;
    Vector<double> weights(diaggmmnormal.weights_);
    for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 4.32 in Dan Povey's thesis.
      weight_auxf_at_start +=
          num_occupancy_(g) * log (weights(g))
          - den_occupancy_(g) * weights(g) / diaggmmnormal.weights_(g);
    }
    for (int32 iter = 0; iter < 50; iter++) {
      Vector<double> k_jm(num_comp); // c.f. eq. 4.35
      double max_m = 0.0;
      for (int32 g = 0; g < num_comp; g++)
        max_m = std::max(max_m, den_occupancy_(g)/diaggmmnormal.weights_(g));
      for (int32 g = 0; g < num_comp; g++)
        k_jm(g) = max_m - den_occupancy_(g)/diaggmmnormal.weights_(g);
      for (int32 g = 0; g < num_comp; g++) // c.f. eq. 4.34
        weights(g) = num_occupancy_(g) + k_jm(g)*weights(g);
      weights.Scale(1.0 / weights.Sum()); // c.f. eq. 4.34 (denominator)
    }
    for (int32 g = 0; g < num_comp; g++) {   // weight flooring.
      if (weights(g) < config.min_gaussian_weight)
        weights(g) = config.min_gaussian_weight;
    }
    weights.Scale(1.0 / weights.Sum()); // renormalize after flooring..
    // floor won't be exact now but doesn't really matter.

    for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 4.32 in Dan Povey's thesis.
      weight_auxf_at_end +=
          num_occupancy_(g) * log (weights(g))
          - den_occupancy_(g) * weights(g) / diaggmmnormal.weights_(g);
    }

    if (auxf_change_out_weights)
      *auxf_change_out_weights += weight_auxf_at_end - weight_auxf_at_start;
    diaggmmnormal.weights_.CopyFromVec(weights);
  }  
  // copy to natural representation according to flags
  diaggmmnormal.CopyToDiagGmm(gmm, flags);
  gmm->ComputeGconsts();
}




MmieAccumDiagGmm::MmieAccumDiagGmm(const MmieAccumDiagGmm &other)
    : dim_(other.dim_), num_comp_(other.num_comp_),
      flags_(other.flags_), num_occupancy_(other.num_occupancy_),
      den_occupancy_(other.den_occupancy_),
      mean_accumulator_(other.mean_accumulator_),
      variance_accumulator_(other.variance_accumulator_) {}



//BaseFloat ComputeD(const DiagGmm& old_gmm, int32 mix_index, BaseFloat ebw_e){
//}



//BaseFloat MmieDiagGmm::MmiObjective(const DiagGmm& gmm) const {
//}

}  // End of namespace kaldi
