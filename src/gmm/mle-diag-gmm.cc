// gmm/mle-diag-gmm.cc

// Copyright 2009-2013  Saarland University;  Georg Stemmer;  Jan Silovsky;
//                      Microsoft Corporation; Yanmin Qian;
//                      Johns Hopkins University (author: Daniel Povey);
//                      Cisco Systems (author: Neha Agrawal)

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

#include <algorithm>  // for std::max
#include <string>
#include <vector>

#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "thread/kaldi-thread.h"

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
                  << dimension << ", " << flags << " (mixing accs from different "
                  << "models?";
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

void AccumDiagGmm::AccumulateForComponent(const VectorBase<BaseFloat> &data,
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
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posteriors) {
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
                                           const VectorBase<BaseFloat> &data,
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
void AccumDiagGmm::SmoothWithAccum(BaseFloat tau, const AccumDiagGmm &src_acc) {
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


void AccumDiagGmm::SmoothWithModel(BaseFloat tau, const DiagGmm &gmm) {
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

BaseFloat MlObjective(const DiagGmm &gmm,
                      const AccumDiagGmm &diag_gmm_acc) {
  GmmFlagsType acc_flags = diag_gmm_acc.Flags();
  Vector<BaseFloat> occ_bf(diag_gmm_acc.occupancy());
  Matrix<BaseFloat> mean_accs_bf(diag_gmm_acc.mean_accumulator());
  Matrix<BaseFloat> variance_accs_bf(diag_gmm_acc.variance_accumulator());
  BaseFloat obj = VecVec(occ_bf, gmm.gconsts());
  if (acc_flags & kGmmMeans)
    obj += TraceMatMat(mean_accs_bf, gmm.means_invvars(), kTrans);
  if (acc_flags & kGmmVariances)
    obj -= 0.5 * TraceMatMat(variance_accs_bf, gmm.inv_vars(), kTrans);
  return obj;
}

void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc,
                      GmmFlagsType flags,
                      DiagGmm *gmm,
                      BaseFloat *obj_change_out,
                      BaseFloat *count_out,
                      int32 *floored_elements_out,
                      int32 *floored_gaussians_out,
                      int32 *removed_gaussians_out) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  KALDI_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
               diag_gmm_acc.Dim() == gmm->Dim());

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().Sum();

  int32 elements_floored = 0, gauss_floored = 0;
  
  // remember old objective value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diag_gmm_acc);

  // First get the gmm in "normal" representation (not the exponential-model
  // form).
  DiagGmmNormal ngmm(*gmm);

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_gauss;

    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
      
      ngmm.weights_(i) = prob;
      
      // copy old mean for later normalizations
      Vector<double> old_mean(ngmm.means_.Row(i));
      
      // update mean, then variance, as far as there are accumulators 
      if (diag_gmm_acc.Flags() & (kGmmMeans|kGmmVariances)) {
        Vector<double> mean(diag_gmm_acc.mean_accumulator().Row(i));
        mean.Scale(1.0 / occ);
        // transfer to estimate
        ngmm.means_.CopyRowFromVec(mean, i);
      }
      
      if (diag_gmm_acc.Flags() & kGmmVariances) {
        KALDI_ASSERT(diag_gmm_acc.Flags() & kGmmMeans);
        Vector<double> var(diag_gmm_acc.variance_accumulator().Row(i));
        var.Scale(1.0 / occ);
        var.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
        
        // if we intend to only update the variances, we need to compensate by 
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          old_mean.AddVec(-1.0, ngmm.means_.Row(i));
          var.AddVec2(1.0, old_mean);
        }
        int32 floored;
        if (config.variance_floor_vector.Dim() != 0) {
          floored = var.ApplyFloor(config.variance_floor_vector);
        } else {
          floored = var.ApplyFloor(config.min_variance);
        }
        if (floored != 0) {
          elements_floored += floored;
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
                   << ", occupation count " << std::fixed << diag_gmm_acc.occupancy()(i)
                   << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
                   << (config.remove_low_count_gaussians ?
                       " it is the last Gaussian: i = "
                       : " remove-low-count-gaussians == false: g = ") << i
                   << ", occ = " << diag_gmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }
  
  // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diag_gmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  if (count_out) *count_out = occ_sum;
  if (floored_elements_out) *floored_elements_out = elements_floored;
  if (floored_gaussians_out) *floored_gaussians_out = gauss_floored;
  
  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights*/);
    gmm->ComputeGconsts();
  }
  if (removed_gaussians_out != NULL) *removed_gaussians_out = to_remove.size();

  if (gauss_floored > 0)
    KALDI_VLOG(2) << gauss_floored << " variances floored in " << gauss_floored
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


void MapDiagGmmUpdate(const MapDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc,
                      GmmFlagsType flags,
                      DiagGmm *gmm,
                      BaseFloat *obj_change_out,
                      BaseFloat *count_out) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  
  KALDI_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
               diag_gmm_acc.Dim() == gmm->Dim());
  
  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().Sum();
  
  // remember the old objective function value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diag_gmm_acc);

  // allocate the gmm in normal representation; all parameters of this will be 
  // updated, but only the flagged ones will be transferred back to gmm
  DiagGmmNormal ngmm(*gmm);

  for (int32 i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_acc.occupancy()(i);

    // First update the weight.  The weight_tau is a tau for the
    // whole state.
    ngmm.weights_(i) = (occ + ngmm.weights_(i) * config.weight_tau) /
        (occ_sum + config.weight_tau);


    if (occ > 0.0 && (flags & kGmmMeans)) {
      // Update the Gaussian mean.
      Vector<double> old_mean(ngmm.means_.Row(i));
      Vector<double> mean(diag_gmm_acc.mean_accumulator().Row(i));
      mean.Scale(1.0 / (occ + config.mean_tau));
      mean.AddVec(config.mean_tau / (occ + config.mean_tau), old_mean);
      ngmm.means_.CopyRowFromVec(mean, i);
    }
    
    if (occ > 0.0 && (flags & kGmmVariances)) {
      // Computing the variance around the updated mean; this is:
      // E( (x - mu)^2 ) = E( x^2 - 2 x mu + mu^2 ) =
      // E(x^2) + mu^2 - 2 mu E(x).
      Vector<double> old_var(ngmm.vars_.Row(i));
      Vector<double> var(diag_gmm_acc.variance_accumulator().Row(i));
      var.Scale(1.0 / occ);
      var.AddVec2(1.0, ngmm.means_.Row(i));
      SubVector<double> mean_acc(diag_gmm_acc.mean_accumulator(), i),
          mean(ngmm.means_, i);
      var.AddVecVec(-2.0 / occ, mean_acc, mean, 1.0);
      // now var is E(x^2) + m^2 - 2 mu E(x).
      // Next we do the appropriate weighting usnig the tau value.
      var.Scale(occ / (config.variance_tau + occ));
      var.AddVec(config.variance_tau / (config.variance_tau + occ), old_var);
      // Now write to the model.
      ngmm.vars_.Row(i).CopyFromVec(var);
    }
  }
  
  // Copy to natural/exponential representation.
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diag_gmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  
  if (count_out) *count_out = occ_sum;
}


class AccumulateMultiThreadedClass: public MultiThreadable {
 public:
  AccumulateMultiThreadedClass(const DiagGmm &diag_gmm,
                               const MatrixBase<BaseFloat> &data,
                               const VectorBase<BaseFloat> &frame_weights,
                               AccumDiagGmm *accum,
                               double *tot_like):
      diag_gmm_(diag_gmm), data_(data),
      frame_weights_(frame_weights), dest_accum_(accum),
      tot_like_ptr_(tot_like), tot_like_(0.0) { }
  AccumulateMultiThreadedClass(const AccumulateMultiThreadedClass &other):
    diag_gmm_(other.diag_gmm_), data_(other.data_),
    frame_weights_(other.frame_weights_), dest_accum_(other.dest_accum_),
    accum_(diag_gmm_, dest_accum_->Flags()), tot_like_ptr_(other.tot_like_ptr_),
    tot_like_(0.0) {
    KALDI_ASSERT(data_.NumRows() == frame_weights_.Dim());
  }
  void operator () () {
    int32 num_frames = data_.NumRows(), num_threads = num_threads_,
        block_size = (num_frames + num_threads - 1) / num_threads,
        block_start = block_size * thread_id_,
        block_end = std::min(num_frames, block_start + block_size);
    tot_like_ = 0.0;
    double tot_weight = 0.0;
    for (int32 t = block_start; t < block_end; t++) {
      tot_like_ += frame_weights_(t) *
          accum_.AccumulateFromDiag(diag_gmm_, data_.Row(t), frame_weights_(t));
      tot_weight += frame_weights_(t);
    }
    KALDI_VLOG(3) << "Thread " << thread_id_ << " saw average likeliood/frame "
                  << (tot_like_ / tot_weight) << " over " << tot_weight
                  << " (weighted) frames.";
  }
  ~AccumulateMultiThreadedClass() {
    if (accum_.Dim() != 0) { // if our accumulator is set up (this is not true
      // for the single object we use to initialize the others)
      dest_accum_->Add(1.0, accum_);
      *tot_like_ptr_ += tot_like_;
    }
  }
 private:
  const DiagGmm &diag_gmm_;
  const MatrixBase<BaseFloat> &data_;
  const VectorBase<BaseFloat> &frame_weights_;
  AccumDiagGmm *dest_accum_;
  AccumDiagGmm accum_;
  double *tot_like_ptr_;
  double tot_like_;
};
  

BaseFloat AccumDiagGmm::AccumulateFromDiagMultiThreaded(
    const DiagGmm &gmm,
    const MatrixBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &frame_weights,
    int32 num_threads) {

  double tot_like = 0.0;
  AccumulateMultiThreadedClass accumulator(gmm, data, frame_weights,
                                           this, &tot_like);
  {
    // Note: everything happens in the constructor and destructor of
    // the object created below.
    MultiThreader<AccumulateMultiThreadedClass> threader(num_threads,
                                                         accumulator);
    // we need to make sure it's destroyed before we access the
    // value of tot_like.
  }
  return tot_like;
}

void AccumDiagGmm::AssertEqual(const AccumDiagGmm &other) {
  KALDI_ASSERT(dim_ == other.dim_ && num_comp_ == other.num_comp_ &&
               flags_ == other.flags_);
  KALDI_ASSERT(occupancy_.ApproxEqual(other.occupancy_));
  KALDI_ASSERT(mean_accumulator_.ApproxEqual(other.mean_accumulator_));
  KALDI_ASSERT(variance_accumulator_.ApproxEqual(other.variance_accumulator_));
}


}  // End of namespace kaldi
