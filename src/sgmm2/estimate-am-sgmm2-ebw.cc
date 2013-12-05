// sgmm2/estimate-am-sgmm2-ebw.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "sgmm2/estimate-am-sgmm2-ebw.h"
#include "thread/kaldi-thread.h"
using std::vector;

namespace kaldi {

void EbwAmSgmm2Updater::Update(const MleAmSgmm2Accs &num_accs,
                               const MleAmSgmm2Accs &den_accs,
                               AmSgmm2 *model,
                               SgmmUpdateFlagsType flags,
                               BaseFloat *auxf_change_out,
                               BaseFloat *count_out) {
  
  // Various quantities need to be computed at the start, before we
  // change any of the model parameters.
  std::vector< SpMatrix<double> > Q_num, Q_den, H, S_means;
  
  if (flags & kSgmmPhoneProjections) {
    MleAmSgmm2Updater::ComputeQ(num_accs, *model, &Q_num);
    MleAmSgmm2Updater::ComputeQ(den_accs, *model, &Q_den);
  }
  if (flags & kSgmmCovarianceMatrix) { // compute the difference between
    // the num and den S_means matrices... this is what we will need.
    MleAmSgmm2Updater::ComputeSMeans(num_accs, *model, &S_means);
    std::vector< SpMatrix<double> > S_means_tmp;
    MleAmSgmm2Updater::ComputeSMeans(den_accs, *model, &S_means_tmp);
    for (size_t i = 0; i < S_means.size(); i++)
      S_means[i].AddSp(-1.0, S_means_tmp[i]);
  }
  if (flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections))
    model->ComputeH(&H);
  
  Vector<double> gamma_num(num_accs.num_gaussians_);
  for (int32 j1 = 0; j1 < num_accs.num_groups_; j1++)
    gamma_num.AddRowSumMat(1.0, num_accs.gamma_[j1]); 
  Vector<double> gamma_den(den_accs.num_gaussians_);
  for (int32 j1 = 0; j1 < den_accs.num_groups_; j1++)
    gamma_den.AddRowSumMat(1.0, den_accs.gamma_[j1]); 

  BaseFloat tot_impr = 0.0;

  if (flags & kSgmmPhoneVectors)
    tot_impr += UpdatePhoneVectors(num_accs, den_accs, H, model);
  
  if (flags & kSgmmPhoneProjections)
    tot_impr += UpdateM(num_accs, den_accs, Q_num, Q_den,
                        gamma_num, gamma_den, model);
  
  if (flags & kSgmmPhoneWeightProjections)
    tot_impr += UpdateW(num_accs, den_accs, gamma_num, gamma_den, model);

  if (flags & kSgmmSpeakerWeightProjections)
    tot_impr += UpdateU(num_accs, den_accs, gamma_num, gamma_den, model);
  
  if (flags & kSgmmCovarianceMatrix)
    tot_impr += UpdateVars(num_accs, den_accs,
                           gamma_num, gamma_den, S_means, model);

  if (flags & kSgmmSubstateWeights)
    tot_impr += UpdateSubstateWeights(num_accs, den_accs, model);

  if (flags & kSgmmSpeakerProjections)
    tot_impr += UpdateN(num_accs, den_accs, gamma_num, gamma_den, model);
  

  if (auxf_change_out) *auxf_change_out = tot_impr * num_accs.total_frames_;
  if (count_out) *count_out = num_accs.total_frames_;
  
  if (fabs(num_accs.total_frames_ - den_accs.total_frames_) >
      0.01*(num_accs.total_frames_ + den_accs.total_frames_))
    KALDI_WARN << "Num and den frame counts differ, "
               << num_accs.total_frames_ << " vs. " << den_accs.total_frames_;

  BaseFloat like_diff = num_accs.total_like_ - den_accs.total_like_;
  
  KALDI_LOG << "***Averaged differenced likelihood per frame is "
            << (like_diff/num_accs.total_frames_)
            << " over " << (num_accs.total_frames_) << " frames.";
  KALDI_LOG << "***Note: for this to be at all meaningful, if you use "
            << "\"canceled\" stats you will have to renormalize this over "
            << "the \"real\" frame count.";
  KALDI_ASSERT(num_accs.total_frames_ > 0 && den_accs.total_frames_ > 0);
  
  model->ComputeNormalizers();
}


class EbwUpdatePhoneVectorsClass: public MultiThreadable { // For multi-threaded.
 public:
  EbwUpdatePhoneVectorsClass(const EbwAmSgmm2Updater *updater,
                             const MleAmSgmm2Accs &num_accs,
                             const MleAmSgmm2Accs &den_accs,
                             const std::vector<SpMatrix<double> > &H,                             
                             AmSgmm2 *model,
                             double *auxf_impr):
      updater_(updater), num_accs_(num_accs), den_accs_(den_accs),
      model_(model), H_(H), auxf_impr_ptr_(auxf_impr), auxf_impr_(0.0) { }
  
  ~EbwUpdatePhoneVectorsClass() {
    *auxf_impr_ptr_ += auxf_impr_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    updater_->UpdatePhoneVectorsInternal(num_accs_, den_accs_, H_, model_,
                                         &auxf_impr_, num_threads_, thread_id_);
  }
 private:
  const EbwAmSgmm2Updater *updater_;
  const MleAmSgmm2Accs &num_accs_;
  const MleAmSgmm2Accs &den_accs_;
  AmSgmm2 *model_;
  const std::vector<SpMatrix<double> > &H_;
  double *auxf_impr_ptr_;
  double auxf_impr_;
};


void EbwAmSgmm2Updater::ComputePhoneVecStats(
    const MleAmSgmm2Accs &accs,
    const AmSgmm2 &model,
    const std::vector<SpMatrix<double> > &H,
    int32 j1,
    int32 m,
    const Vector<double> &w_jm_in,
    double gamma_jm,
    Vector<double> *g_jm,
    SpMatrix<double> *H_jm) {
  Vector<double> w_jm(w_jm_in);
  if (!accs.a_.empty() && accs.a_[j1](m, 0) != 0) { // [SSGMM]
    w_jm.MulElements(accs.a_[j1].Row(m)); // multiply by "a" quantities..
    w_jm.Scale(1.0 / w_jm.Sum()); // renormalize.
  }  
  g_jm->CopyFromVec(accs.y_[j1].Row(m));
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    double gamma_jmi = accs.gamma_[j1](m, i);
    double quadratic_term = std::max(gamma_jmi, gamma_jm * w_jm(i));
    double scalar = gamma_jmi - gamma_jm * w_jm(i) + quadratic_term
        * VecVec(model.w_.Row(i), model.v_[j1].Row(m));
    g_jm->AddVec(scalar, model.w_.Row(i));
    if (gamma_jmi != 0.0)
      H_jm->AddSp(gamma_jmi, H[i]);  // The most important term..
    if (quadratic_term > 1.0e-10)
      H_jm->AddVec2(static_cast<BaseFloat>(quadratic_term), model.w_.Row(i));
  }
}


// Runs the phone vectors update for a subset of states (called
// multi-threaded).
void EbwAmSgmm2Updater::UpdatePhoneVectorsInternal(
    const MleAmSgmm2Accs &num_accs,
    const MleAmSgmm2Accs &den_accs,
    const std::vector<SpMatrix<double> > &H,
    AmSgmm2 *model,
    double *auxf_impr,
    int32 num_threads,
    int32 thread_id) const {

  int32 block_size = (num_accs.num_groups_ + (num_threads-1)) / num_threads,
      j1_start = block_size * thread_id,
      j1_end = std::min(num_accs.num_groups_, j1_start + block_size);
  
  int32 S = num_accs.phn_space_dim_, I = num_accs.num_gaussians_;
  
  for (int32 j1 = j1_start; j1 < j1_end; j1++) {
    double num_state_count = 0.0,
        state_auxf_impr = 0.0;
    Vector<double> w_jm(I);
    for (int32 m = 0; m < model->NumSubstatesForGroup(j1); m++) {
      double gamma_jm_num = num_accs.gamma_[j1].Row(m).Sum();
      double gamma_jm_den = den_accs.gamma_[j1].Row(m).Sum();
      num_state_count += gamma_jm_num;
      Vector<double> g_jm_num(S);  // computed using eq. 58 of SGMM paper [for numerator stats]
      SpMatrix<double> H_jm_num(S);  // computed using eq. 59 of SGMM paper [for numerator stats]
      Vector<double> g_jm_den(S); // same, but for denominator stats.
      SpMatrix<double> H_jm_den(S);

      // Compute the weights for this sub-state.
      // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
      w_jm.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
                     Vector<double>(model->v_[j1].Row(m)), 0.0);
      w_jm.ApplySoftMax();
      // Note: in the ML code, in the SSGMM case, at this point the w_jm would
      // be modified with the "a" quantities to get the "\tilde{w}_{jm}" of the
      // SSGMM techreport.  But in this code, it gets done inside ComputePhoneVecStats.
      
      ComputePhoneVecStats(num_accs, *model, H, j1, m, w_jm, gamma_jm_num,
                           &g_jm_num, &H_jm_num);
      ComputePhoneVecStats(den_accs, *model, H, j1, m, w_jm, gamma_jm_den,
                           &g_jm_den, &H_jm_den);
      
      Vector<double> v_jm(model->v_[j1].Row(m));
      Vector<double> local_derivative(S); // difference of derivative of numerator
      // and denominator objetive function.
      local_derivative.AddVec(1.0, g_jm_num);
      local_derivative.AddSpVec(-1.0, H_jm_num, v_jm, 1.0);
      local_derivative.AddVec(-1.0, g_jm_den);
      local_derivative.AddSpVec(-1.0 * -1.0, H_jm_den, v_jm, 1.0);
      
      SpMatrix<double> quadratic_term(H_jm_num);
      quadratic_term.AddSp(1.0, H_jm_den);
      double substate_count = 1.0e-10 + gamma_jm_num + gamma_jm_den;
      quadratic_term.Scale( (substate_count + options_.tau_v) / substate_count);
      quadratic_term.Scale(1.0 / (options_.lrate_v + 1.0e-10) );

      Vector<double> delta_v_jm(S);

      SolverOptions opts;
      opts.name = "v";
      opts.K = options_.max_cond;
      opts.eps = options_.epsilon;
      
      double auxf_impr =
          ((gamma_jm_num + gamma_jm_den == 0) ? 0.0 :
           SolveQuadraticProblem(quadratic_term,
                                 local_derivative,
                                 opts, &delta_v_jm));

      v_jm.AddVec(1.0, delta_v_jm);
      model->v_[j1].Row(m).CopyFromVec(v_jm);
      state_auxf_impr += auxf_impr;
    }

    *auxf_impr += state_auxf_impr;
    if (j1 < 10 && thread_id == 0) {
      KALDI_LOG << "Objf impr for group j = " << j1 << "  is "
                << (state_auxf_impr / (num_state_count + 1.0e-10))
                << " over " << num_state_count << " frames";
    }
  }
}

double EbwAmSgmm2Updater::UpdatePhoneVectors(const MleAmSgmm2Accs &num_accs,
                                             const MleAmSgmm2Accs &den_accs,
                                             const vector< SpMatrix<double> > &H,
                                             AmSgmm2 *model) const {
  KALDI_LOG << "Updating phone vectors.";
  
  double count = 0.0, auxf_impr = 0.0;
  
  int32 J1 = num_accs.num_groups_;
  for (int32 j1 = 0; j1 < J1; j1++) count += num_accs.gamma_[j1].Sum();
  
  EbwUpdatePhoneVectorsClass c(this, num_accs, den_accs, H, model, &auxf_impr);
  RunMultiThreaded(c);

  auxf_impr /= count;

  KALDI_LOG << "**Overall auxf improvement for v is " << auxf_impr
            << " over " << count << " frames";
  return auxf_impr;
}


double EbwAmSgmm2Updater::UpdateM(const MleAmSgmm2Accs &num_accs,
                                  const MleAmSgmm2Accs &den_accs,
                                  const std::vector< SpMatrix<double> > &Q_num,
                                  const std::vector< SpMatrix<double> > &Q_den,
                                  const Vector<double> &gamma_num,
                                  const Vector<double> &gamma_den,
                                  AmSgmm2 *model) const {
  int32 S = model->PhoneSpaceDim(),
      D = model->FeatureDim(),
      I = model->NumGauss();
  
  Vector<double> impr_vec(I);

  for (int32 i = 0; i < I; i++) {
    double gamma_i_num = gamma_num(i), gamma_i_den = gamma_den(i);

    if (gamma_i_num + gamma_i_den == 0.0) {
      KALDI_WARN << "Not updating phonetic basis for i = " << i
                 << " because count is zero. ";
      continue;
    }
    
    Matrix<double> Mi(model->M_[i]);    
    Matrix<double> L(D, S); // this is something like the Y quantity, which
    // represents the linear term in the objf on M-- except that we make it the local
    // derivative about the current value, instead of the derivative around zero.
    // But it's not exactly the derivative w.r.t. M, due to the factor of Sigma_i.
    // The auxiliary function is Q(x) = tr(M^T P Y) - 0.5 tr(P M Q M^T),
    // where P is Y^{-1}.  The quantity L we define here will be Y - M Q,
    // and you can think of this as like the local derivative, except there is
    // a term P in there.
    L.AddMat(1.0, num_accs.Y_[i]);
    L.AddMatSp(-1.0, Mi, kNoTrans, Q_num[i], 1.0);
    L.AddMat(-1.0, den_accs.Y_[i]);
    L.AddMatSp(-1.0*-1.0, Mi, kNoTrans, Q_den[i], 1.0);

    SpMatrix<double> Q(S); // This is a combination of the Q's for the numerator and denominator.
    Q.AddSp(1.0, Q_num[i]);
    Q.AddSp(1.0, Q_den[i]);

    double state_count = 1.0e-10 + gamma_i_num + gamma_i_den; // the count
    // represented by the quadratic part of the stats.
    Q.Scale( (state_count + options_.tau_M) / state_count );
    Q.Scale( 1.0 / (options_.lrate_M + 1.0e-10) );
    

    SolverOptions opts;
    opts.name = "M";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;

    Matrix<double> deltaM(D, S);
    double impr =
        SolveQuadraticMatrixProblem(Q, L,
                                    SpMatrix<double>(model->SigmaInv_[i]),
                                    opts, &deltaM);

    impr_vec(i) = impr;
    Mi.AddMat(1.0, deltaM);
    model->M_[i].CopyFromMat(Mi);
    if (i < 10 || impr / state_count > 3.0) {
      KALDI_VLOG(2) << "Objf impr for projection M for i = " << i << ", is "
                    << (impr/(gamma_i_num + 1.0e-20)) << " over " << gamma_i_num
                    << " frames";
    }
  }
  BaseFloat tot_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  
  tot_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "Overall auxiliary function improvement for model projections "
            << "M is " << tot_impr << " over " << tot_count << " frames";

  KALDI_VLOG(1) << "Updating M: num-count is " << gamma_num;
  KALDI_VLOG(1) << "Updating M: den-count is " << gamma_den;
  KALDI_VLOG(1) << "Updating M: objf-impr is " << impr_vec;
  
  return tot_impr;
}


// Note: we do just one iteration of the weight-projection update here.  The
// weak-sense auxiliary functions used don't really make sense if we do it for
// multiple iterations.  It would be possible to use a similar auxiliary
// function to the one on my (D. Povey)'s thesis for the Gaussian mixture
// weights, which would make sense for multiple iterations, but this would be a
// bit more complex to implement and probably would not give much improvement
// over this approach.
double EbwAmSgmm2Updater::UpdateW(const MleAmSgmm2Accs &num_accs,
                                  const MleAmSgmm2Accs &den_accs,
                                  const Vector<double> &gamma_num,
                                  const Vector<double> &gamma_den,
                                  AmSgmm2 *model) {
  KALDI_LOG << "Updating weight projections";

  int32 I = num_accs.num_gaussians_, S = num_accs.phn_space_dim_;
  
  Matrix<double> g_i_num(I, S), g_i_den(I, S);
      
  // View F_i_{num,den} as vectors of SpMatrix [i.e. symmetric matrices,
  // linearized into vectors]
  Matrix<double> F_i_num(I, (S*(S+1))/2), F_i_den(I, (S*(S+1))/2);
  
  Vector<double> impr_vec(I);
  
  // Get the F_i and g_i quantities-- this is done in parallel (multi-core),
  // using the same code we use in the ML update [except we get it for
  // numerator and denominator separately.]
  Matrix<double> w(model->w_);
  {
    std::vector<Matrix<double> > log_a_num;
    if (model->HasSpeakerDependentWeights())
      MleAmSgmm2Updater::ComputeLogA(num_accs, &log_a_num);
    double garbage;
    UpdateWClass c_num(num_accs, *model, w, log_a_num, &F_i_num, &g_i_num, &garbage);
    RunMultiThreaded(c_num);
  }
  {
    std::vector<Matrix<double> > log_a_den;
    if (model->HasSpeakerDependentWeights())    
      MleAmSgmm2Updater::ComputeLogA(den_accs, &log_a_den);
    double garbage;
    UpdateWClass c_den(den_accs, *model, w, log_a_den, &F_i_den, &g_i_den, &garbage);
    RunMultiThreaded(c_den);
  }

  for (int32 i = 0; i < I; i++) {

    // auxf was originally formulated in terms of the change in w (i.e. the
    // g quantities are the local derivatives), so there is less hassle than
    // with some of the other updates, in changing it to be discriminative.
    // we essentially just difference the linear terms and add the quadratic
    // terms.

    Vector<double> derivative(g_i_num.Row(i));
    derivative.AddVec(-1.0, g_i_den.Row(i));
    // F_i_num quadratic_term is a bit like the negated 2nd derivative
    // of the numerator stats-- actually it's not the actual 2nd deriv,
    // but an upper bound on it.
    SpMatrix<double> quadratic_term(S), tmp_F(S);
    quadratic_term.CopyFromVec(F_i_num.Row(i));
    tmp_F.CopyFromVec(F_i_den.Row(i)); // tmp_F is used for Vector->SpMatrix conversion.
    quadratic_term.AddSp(1.0, tmp_F);

    double state_count = gamma_num(i) + gamma_den(i);

    quadratic_term.Scale((state_count + options_.tau_w) / (state_count + 1.0e-10));
    quadratic_term.Scale(1.0 / (options_.lrate_w + 1.0e-10) );
    
    Vector<double> delta_w(S);

    SolverOptions opts;
    opts.name = "w";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
  
    double objf_impr =
        SolveQuadraticProblem(quadratic_term, derivative, opts, &delta_w);

    impr_vec(i) = objf_impr;
    if (i < 10 || objf_impr / (gamma_num(i) + 1.0e-10) > 2.0) {
      KALDI_LOG << "Predicted objf impr for w per frame is "
                << (objf_impr / (gamma_num(i) + 1.0e-10))
                << " over " << gamma_num(i) << " frames.";
    }
    model->w_.Row(i).AddVec(1.0, delta_w);
  }
  KALDI_VLOG(1) << "Updating w: numerator count is " << gamma_num;
  KALDI_VLOG(1) << "Updating w: denominator count is " << gamma_den;
  KALDI_VLOG(1) << "Updating w: objf-impr is " << impr_vec;
    
  double tot_num_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  tot_impr /= tot_num_count;

  KALDI_LOG << "**Overall objf impr for w per frame is "
            << tot_impr << " over " << tot_num_count
            << " frames.";
  return tot_impr;
}


double EbwAmSgmm2Updater::UpdateU(const MleAmSgmm2Accs &num_accs,
                                  const MleAmSgmm2Accs &den_accs,
                                  const Vector<double> &gamma_num,
                                  const Vector<double> &gamma_den,
                                  AmSgmm2 *model) {
  int32 T = num_accs.spk_space_dim_;
  double tot_impr = 0.0;
  for (int32 i = 0; i < num_accs.num_gaussians_; i++) {
    if (gamma_num(i) < 200.0) {
      KALDI_LOG << "Numerator count is small " << gamma_num(i) << " for gaussian "
                << i << ", not updating u_i.";
      continue;
    }
    Vector<double> u_i(model->u_.Row(i));
    Vector<double> delta_u(T);
    Vector<double> t(T); // derivative.
    t.AddVec(1.0, num_accs.t_.Row(i));
    t.AddVec(-1.0, den_accs.t_.Row(i));
    SpMatrix<double> U(T); // quadratic term.
    U.AddSp(1.0, num_accs.U_[i]);
    U.AddSp(1.0, den_accs.U_[i]);

    double state_count = gamma_num(i) + gamma_den(i);
    U.Scale((state_count + options_.tau_u) / (state_count + 1.0e-10));
    U.Scale(1.0 / (options_.lrate_u + 1.0e-10) );
    
    SolverOptions opts;
    opts.name = "u";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
  
    double impr = SolveQuadraticProblem(U, t, opts, &delta_u);
    double impr_per_frame = impr / gamma_num(i);
    if (impr_per_frame > options_.max_impr_u) {
      KALDI_WARN << "Updating speaker weight projections u, for Gaussian index "
                 << i << ", impr/frame is " << impr_per_frame << " over "
                 << gamma_num(i) << " frames, scaling back to not exceed "
                 << options_.max_impr_u;
      double scale = options_.max_impr_u / impr_per_frame;
      impr *= scale;
      delta_u.Scale(scale);
      // Note: a linear scaling of "impr" with "scale" is not quite accurate
      // in depicting how the quadratic auxiliary function varies as we change
      // the scale on "delta", but this does not really matter-- the goal is
      // to limit the auxiliary-function change to not be too large.
    }
    if (i < 10) {
      KALDI_LOG << "Objf impr for spk weight-projection u for i = " << (i)
                << ", is " << (impr / (gamma_num(i) + 1.0e-20)) << " over "
                << gamma_num(i) << " frames";
    }
    u_i.AddVec(1.0, delta_u);
    model->u_.Row(i).CopyFromVec(u_i);
    tot_impr += impr;
  }
  KALDI_LOG << "**Overall objf impr for u is " << (tot_impr/gamma_num.Sum())
            << ", over " << gamma_num.Sum() << " frames";
  return tot_impr;
}


double EbwAmSgmm2Updater::UpdateN(const MleAmSgmm2Accs &num_accs,
                                  const MleAmSgmm2Accs &den_accs,
                                  const Vector<double> &gamma_num,
                                  const Vector<double> &gamma_den,
                                  AmSgmm2 *model) const {
  if (num_accs.spk_space_dim_ == 0 || num_accs.R_.size() == 0 ||
      num_accs.Z_.size() == 0) {
    KALDI_ERR << "Speaker subspace dim is zero or no stats accumulated";
  }

  int32 I = num_accs.num_gaussians_, D = num_accs.feature_dim_,
      T = num_accs.spk_space_dim_;
  
  Vector<double> impr_vec(I);
  
  for (int32 i = 0; i < I; i++) {
    double gamma_i_num = gamma_num(i), gamma_i_den = gamma_den(i);
    if (gamma_i_num + gamma_i_den == 0.0) {
      KALDI_WARN << "Not updating speaker basis for i = " << i
                 << " because count is zero. ";
      continue;
    }
    Matrix<double> Ni(model->N_[i]);
    // See comment near declaration of L in UpdateM().  This update is the
    // same, but change M->N, Y->Z and Q->R.

    Matrix<double> L(D, T);
    L.AddMat(1.0, num_accs.Z_[i]);
    L.AddMatSp(-1.0, Ni, kNoTrans, num_accs.R_[i], 1.0);
    L.AddMat(-1.0, den_accs.Z_[i]);
    L.AddMatSp(-1.0*-1.0, Ni, kNoTrans, den_accs.R_[i], 1.0);
    
    SpMatrix<double> R(T); // combination of the numerator and denominator R's.
    R.AddSp(1.0, num_accs.R_[i]);
    R.AddSp(1.0, den_accs.R_[i]);

    double state_count = 1.0e-10 + gamma_i_num + gamma_i_den; // the count
    // represented by the quadratic part of the stats.
    R.Scale( (state_count + options_.tau_N) / state_count );
    R.Scale( 1.0 / (options_.lrate_N + 1.0e-10) );
    
    Matrix<double> deltaN(D, T);

    SolverOptions opts;
    opts.name = "N";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
  
    double impr =
        SolveQuadraticMatrixProblem(R, L,
                                    SpMatrix<double>(model->SigmaInv_[i]),
                                    opts, &deltaN);
    impr_vec(i) = impr;
    Ni.AddMat(1.0, deltaN);
    model->N_[i].CopyFromMat(Ni);
    if (i < 10 || impr / (state_count+1.0e-20) > 3.0) {
      KALDI_LOG << "Objf impr for spk projection N for i = " << (i)
                << ", is " << (impr / (gamma_i_num + 1.0e-20)) << " over "
                << gamma_i_num << " frames";
    }
  }

  KALDI_VLOG(1) << "Updating N: numerator count is " << gamma_num;
  KALDI_VLOG(1) << "Updating N: denominator count is " << gamma_den;
  KALDI_VLOG(1) << "Updating N: objf-impr is " << impr_vec;
  
  double tot_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  tot_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "**Overall auxf impr for N is " << tot_impr
            << " over " << tot_count << " frames";
  return tot_impr;
}

double EbwAmSgmm2Updater::UpdateVars(const MleAmSgmm2Accs &num_accs,
                                     const MleAmSgmm2Accs &den_accs,
                                     const Vector<double> &gamma_num,
                                     const Vector<double> &gamma_den,
                                     const std::vector< SpMatrix<double> > &S_means,
                                     AmSgmm2 *model) const {
  // Note: S_means contains not only the quantity S_means in the paper,
  // but also has a term - (Y_i M_i^T + M_i Y_i^T).  Plus, it is differenced
  // between numerator and denominator.  We don't calculate it here,
  // because it had to be computed with the original model, before we
  // changed the M quantities.
  int32 I = num_accs.num_gaussians_;
  KALDI_ASSERT(S_means.size() == I);
  Vector<double> impr_vec(I);
  
  for (int32 i = 0; i < I; i++) {
    double num_count = gamma_num(i), den_count = gamma_den(i);

    SpMatrix<double> SigmaStats(S_means[i]);
    SigmaStats.AddSp(1.0, num_accs.S_[i]);
    SigmaStats.AddSp(-1.0, den_accs.S_[i]);
    // SigmaStats now contain the stats for estimating Sigma (as in the main SGMM paper),
    // differenced between num and den.
    SpMatrix<double> SigmaInvOld(model->SigmaInv_[i]), SigmaOld(model->SigmaInv_[i]);
    SigmaOld.Invert();
    double count = num_count - den_count;
    KALDI_ASSERT(options_.lrate_Sigma <= 1.0);
    double inv_lrate = 1.0 / options_.lrate_Sigma;
    // These formulas assure that the objective function behaves in
    // a roughly symmetric way w.r.t. num and den counts.
    double E_den = 1.0 + inv_lrate, E_num = inv_lrate - 1.0;

    double smoothing_count =
        (options_.tau_Sigma * inv_lrate) + // multiply tau_Sigma by inverse-lrate
        (E_den * den_count) +              // for compatibility with other updates.
        (E_num * num_count) +
        1.0e-10;
    SigmaStats.AddSp(smoothing_count, SigmaOld);
    count += smoothing_count;
    SigmaStats.Scale(1.0 / count);
    SpMatrix<double> SigmaInv(SigmaStats); // before floor and ceiling.  Currently sigma,
    // not its inverse.
    bool verbose = false;
    bool is_psd = false; // we cannot guarantee that Sigma Inv is positive semidefinite.
    int n_floor = SigmaInv.ApplyFloor(SigmaOld, options_.cov_min_value, verbose, is_psd);
    SigmaInv.Invert(); // make it inverse variance.
    int n_ceiling = SigmaInv.ApplyFloor(SigmaInvOld, options_.cov_min_value, verbose, is_psd);
    
    // this auxf_change.  
    double auxf_change = -0.5 * count *(TraceSpSp(SigmaInv, SigmaStats)
                                        - TraceSpSp(SigmaInvOld, SigmaStats)
                                        - SigmaInv.LogDet()
                                        + SigmaInvOld.LogDet());

    model->SigmaInv_[i].CopyFromSp(SigmaInv);
    impr_vec(i) = auxf_change;
    if (i < 10 || auxf_change / (num_count+den_count+1.0e-10) > 2.0
        || n_floor+n_ceiling > 0) {
      KALDI_LOG << "Updating variance: Auxf change per frame for Gaussian "
                << i << " is " << (auxf_change / num_count) << " over "
                << num_count << " frames " << "(den count was " << den_count
                << "), #floor,ceil was " << n_floor << ", " << n_ceiling;
    }
  }
  KALDI_VLOG(1) << "Updating Sigma: numerator count is " << gamma_num;
  KALDI_VLOG(1) << "Updating Sigma: denominator count is " << gamma_den;
  KALDI_VLOG(1) << "Updating Sigma: objf-impr is " << impr_vec;
  
  double tot_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  tot_impr /= tot_count+1.0e-20;
  KALDI_LOG << "**Overall auxf impr for Sigma is " << tot_impr
            << " over " << tot_count << " frames";
  return tot_impr;
}


double EbwAmSgmm2Updater::UpdateSubstateWeights(
    const MleAmSgmm2Accs &num_accs,
    const MleAmSgmm2Accs &den_accs,
    AmSgmm2 *model) {
  KALDI_LOG << "Updating substate mixture weights";

  double tot_count = 0.0, tot_impr = 0.0;
  for (int32 j2 = 0; j2 < num_accs.num_pdfs_; j2++) {
    int32 M = model->NumSubstatesForPdf(j2);
    Vector<double> num_occs(M), den_occs(M),
        orig_weights(model->c_[j2]), weights(model->c_[j2]);

    for (int32 m = 0; m < M; m++) {
      num_occs(m) = num_accs.gamma_c_[j2](m)
          + options_.tau_c * weights(m);
      den_occs(m) = den_accs.gamma_c_[j2](m);
    }
    
    if (weights.Dim() > 1) {
      double begin_auxf = 0.0, end_auxf = 0.0;
      for (int32 m = 0; m < M; m++) {  // see eq. 4.32, Dan Povey's PhD thesis.
        begin_auxf += num_occs(m) * log (weights(m))
            - den_occs(m) * weights(m) / orig_weights(m);
      }
      for (int32 iter = 0; iter < 50; iter++) {
        Vector<double> k_jm(M);
        double max_m = 0.0;
        for (int32 m = 0; m < M; m++)
          max_m = std::max(max_m, den_occs(m)/orig_weights(m));
        for (int32 m = 0; m < M; m++)
          k_jm(m) = max_m - den_occs(m)/orig_weights(m);
        for (int32 m = 0; m < M; m++)
          weights(m) = num_occs(m) + k_jm(m)*weights(m);
        weights.Scale(1.0 / weights.Sum());
      }
      for (int32 m = 0; m < M; m++)
        weights(m) = std::max(weights(m),
                              static_cast<double>(options_.min_substate_weight));
      weights.Scale(1.0 / weights.Sum()); // renormalize.

      for (int32 m = 0; m < M; m++) {
        end_auxf += num_occs(m) * log (weights(m))
            - den_occs(m) * weights(m) / orig_weights(m);
      }
      tot_impr += end_auxf - begin_auxf;
      double this_impr = ((end_auxf - begin_auxf) / num_occs.Sum());
      if (j2 < 10 || this_impr > 0.5) {
        KALDI_LOG << "Updating substate weights: auxf impr for pdf " << j2
                  << " is " << this_impr << " per frame over " << num_occs.Sum()
                  << " frames (den count is " << den_occs.Sum() << ")";
      }
    }
    model->c_[j2].CopyFromVec(weights);
    tot_count += den_occs.Sum(); // Note: num and den occs should be the
    // same, except num occs are smoothed, so this is what we want.
  }    

  tot_impr /= (tot_count + 1.0e-20);

  KALDI_LOG << "**Overall auxf impr for c is " << tot_impr
            << " over " << tot_count << " frames";
  return tot_impr;
}  

}  // namespace kaldi
