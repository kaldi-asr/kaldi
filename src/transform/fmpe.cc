// transform/fmpe.cc

// Copyright 2011-2012  Yanmin Qian  Daniel Povey

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


#include "transform/fmpe.h"
#include "util/text-utils.h"
#include "gmm/diag-gmm-normal.h"

namespace kaldi {

void Fmpe::SetContexts(std::string context_str) {
  // sets the contexts_ variable.
  using std::vector;
  using std::string;
  contexts_.clear();
  vector<string> ctx_vec; // splitting context_str on ":"
  SplitStringToVector(context_str, ":", &ctx_vec, false);
  contexts_.resize(ctx_vec.size());
  for (size_t i = 0; i < ctx_vec.size(); i++) {
    vector<string> pair_vec; // splitting ctx_vec[i] on ";"
    SplitStringToVector(ctx_vec[i], ";", &pair_vec, false);
    KALDI_ASSERT(pair_vec.size() != 0 && "empty context!");
    for (size_t j = 0; j < pair_vec.size(); j++) {
      vector<string> one_pair;
      SplitStringToVector(pair_vec[j], ",", &one_pair, false);
      KALDI_ASSERT(one_pair.size() == 2 &&
                   "Mal-formed context string: bad --context-expansion option?");
      int32 pos;
      BaseFloat weight;
      if (!ConvertStringToInteger(one_pair[0], &pos)
          || !ConvertStringToReal(one_pair[1], &weight))
        KALDI_ERR << "Mal-formed context string: bad --context-expansion option?";
      contexts_[i].push_back(std::make_pair(pos, weight));
    }
  }
}

void Fmpe::ComputeC() {
  KALDI_ASSERT(gmm_.NumGauss() != 0.0);
  int32 dim = gmm_.Dim();

  // Getting stats from the GMM... assume the model is
  // correct.
  SpMatrix<double> x2_stats(dim);
  Vector<double> x_stats(dim);
  double tot_count = 0.0;
  DiagGmmNormal ngmm(gmm_);
  for (int32 pdf = 0; pdf < ngmm.NumGauss(); pdf++) {
    x2_stats.AddVec2(ngmm.weights_(pdf), ngmm.means_.Row(pdf));
    x2_stats.AddVec(ngmm.weights_(pdf), ngmm.vars_.Row(pdf)); // add diagonal
    // covar to diagonal elements of x2_stats.
    x_stats.AddVec(ngmm.weights_(pdf), ngmm.means_.Row(pdf));
    tot_count += ngmm.weights_(pdf);
  }
  KALDI_ASSERT(tot_count != 0.0);
  x2_stats.Scale(1.0 / tot_count);
  x_stats.Scale(1.0 / tot_count);
  x2_stats.AddVec2(-1.0, x_stats); // subtract outer product of mean,
  // to get centered covariance.
  C_.Resize(dim);
  try {
    TpMatrix<double> Ctmp; Ctmp.Cholesky(x2_stats);
    C_.CopyFromTp(Ctmp);
  } catch (...) {
    KALDI_ERR << "Error initializing fMPE object: cholesky of "
        "feature variance failed.  Probably code error, or NaN/inf in model";
  }
}

void Fmpe::ComputeStddevs() {
  const Matrix<BaseFloat> &inv_vars = gmm_.inv_vars();
  stddevs_.Resize(inv_vars.NumRows(), inv_vars.NumCols());
  stddevs_.CopyFromMat(inv_vars);
  stddevs_.ApplyPow(-0.5);
}


void Fmpe::ApplyContext(const MatrixBase<BaseFloat> &intermed_feat,
                        MatrixBase<BaseFloat> *feat_out) const {
  // Applies the temporal-context part of the transformation.
  int32 dim = FeatDim(), ncontexts = NumContexts(),
      T = intermed_feat.NumRows();
  KALDI_ASSERT(intermed_feat.NumRows() == dim * ncontexts &&
               intermed_feat.NumCols() == feat_out->NumCols()
               && feat_out->NumRows() == dim);
  // note: ncontexts == contexts_.size().
  for (int32 i = 0; i < ncontexts; i++) {
    // this_intermed_feat is the chunk of the "intermediate features"
    // that corresponds to this "context"
    SubMatrix<BaseFloat> this_intermed_feat(intermed_feat, 0, T,
                                            dim*i, dim);
    for (int32 j = 0; j < static_cast<int32>(contexts_[i].size()); j++) {
      int32 t_offset = contexts_[i][j].first;
      BaseFloat weight = contexts_[i][j].second;
      // Note: we could do this more efficiently using matrix operations,
      // but this doesn't dominate the computation and I think this is
      // clearer.
      for (int32 t_out = 0; t_out < T; t_out++) { // t_out indexes the output
        int32 t_in = t_out + t_offset; // t_in indexes the input.
        if (t_in >= 0 && t_in < T) // Discard frames outside range.
          feat_out->Row(t_out).AddVec(weight, this_intermed_feat.Row(t_in));
      }
    }
  }
}

void Fmpe::ApplyContextReverse(const MatrixBase<BaseFloat> &feat_deriv,
                               MatrixBase<BaseFloat> *intermed_feat_deriv)
    const {
  // Applies the temporal-context part of the transformation,
  // in reverse, for getting derivatives for training.
  int32 dim = FeatDim(), ncontexts = NumContexts(),
      T = feat_deriv.NumRows();
  KALDI_ASSERT(intermed_feat_deriv->NumRows() == dim * ncontexts &&
               intermed_feat_deriv->NumCols() == feat_deriv.NumCols()
               && feat_deriv.NumRows() == dim);
  // note: ncontexts == contexts_.size().
  for (int32 i = 0; i < ncontexts; i++) {
    // this_intermed_feat is the chunk of the derivative of
    // "intermediate features" that corresponds to this "context"
    // (this is output, in this routine).
    SubMatrix<BaseFloat> this_intermed_feat_deriv(*intermed_feat_deriv, 0, T,
                                                  dim*i, dim);
    for (int32 j = 0; j < static_cast<int32>(contexts_[i].size()); j++) {
      int32 t_offset = contexts_[i][j].first;
      BaseFloat weight = contexts_[i][j].second;
      // Note: we could do this more efficiently using matrix operations,
      // but this doesn't dominate the computation and I think this is
      // clearer.
      for (int32 t_out = 0; t_out < T; t_out++) { // t_out indexes the output
        int32 t_in = t_in + t_offset; // t_in indexes the input.
        if (t_in >= 0 && t_in < T) // Discard frames outside range.
          this_intermed_feat_deriv.Row(t_in).AddVec(weight,
                                                    feat_deriv.Row(t_out));
        // Note: the line above is where the work happens; it's the same
        // as in ApplyContext except reversing the input and output.
      }
    }
  }
}

void Fmpe::ApplyC(MatrixBase<BaseFloat> *feat_out, bool reverse) const {
  int32 T = feat_out->NumRows();
  Vector<BaseFloat> tmp(feat_out->NumCols());
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> row(*feat_out, t);
    // Next line does: tmp = C_ * row
    tmp.AddTpVec(1.0, C_, (reverse ? kTrans : kNoTrans), row, 0.0);
    row.CopyFromVec(tmp);
  }
}

// Constructs the high-dim features and applies the main projection matrix proj_.
void Fmpe::ApplyProjection(const MatrixBase<BaseFloat> &feat_in,
                           const std::vector<std::vector<int32> > &gselect,
                           MatrixBase<BaseFloat> *intermed_feat) const {
  int32 dim = FeatDim(), ncontexts = NumContexts();  
  
  Vector<BaseFloat> post; // will be posteriors of selected Gaussians.
  Vector<BaseFloat> input_chunk(dim+1); // will be a segment of
  // the high-dimensional features.
  for (int32 t = 0; t < feat_in.NumRows(); t++) {
    SubVector<BaseFloat> this_feat(feat_in, t);
    SubVector<BaseFloat> this_intermed_feat(*intermed_feat, t);
    gmm_.LogLikelihoodsPreselect(this_feat, gselect[t], &post);
    // At this point, post will contain log-likes of the selected
    // Gaussians.
    post.ApplySoftMax(); // Now they are posteriors (which sum to one).
    for (int32 i = 0; i < post.Dim(); i++) {
      int32 gauss = gselect[t][i];
      SubVector<BaseFloat> this_stddev(stddevs_, gauss);
      BaseFloat this_post = post(i);
      // The next line is equivalent to setting input_chunk to
      // -this_post * the gaussian mean / (gaussian stddev).  Note: we use
      // the fact that mean * inv_var *  stddev == mean / stddev.
      input_chunk.Range(0, dim).AddVecVec(-this_post, gmm_.means_invvars().Row(gauss),
                                          this_stddev, 0.0);
      // The next line is equivalent to adding (feat / gaussian stddev) to
      // input_chunk, so now it contains (feat - mean) / stddev, which is
      // our "normalized" feature offset.
      input_chunk.Range(0, dim).AddVecDivVec(this_post, this_feat, this_stddev,
                                             1.0);
      // The last element of this input_chunk is the posterior itself
      // (between 0 and 1).
      input_chunk(dim) = this_post;

      // this_intermed_feat += [appropriate chjunk of proj_] * input_chunk.
      this_intermed_feat.AddMatVec(1.0, proj_.Range(0, dim*ncontexts,
                                                    gauss*(dim+1), dim+1),
                                   kNoTrans, input_chunk, 1.0);
    }
  }
}      



// This function does the reverse to ApplyProjection, for the case
// where we want the derivatives w.r.t. the projection matrix.
// It stores the positive and negative parts of this separately.
void Fmpe::ApplyProjectionReverse(const MatrixBase<BaseFloat> &feat_in,
                                  const std::vector<std::vector<int32> > &gselect,
                                  const MatrixBase<BaseFloat> &intermed_feat_deriv,
                                  MatrixBase<BaseFloat> *proj_deriv_plus,
                                  MatrixBase<BaseFloat> *proj_deriv_minus) const {
  int32 dim = FeatDim(), ncontexts = NumContexts();  
  
  Vector<BaseFloat> post; // will be posteriors of selected Gaussians.
  Vector<BaseFloat> input_chunk(dim+1); // will be a segment of
  // the high-dimensional features.
  for (int32 t = 0; t < feat_in.NumRows(); t++) {
    SubVector<BaseFloat> this_feat(feat_in, t);
    SubVector<BaseFloat> this_intermed_feat_deriv(intermed_feat_deriv, t);
    gmm_.LogLikelihoodsPreselect(this_feat, gselect[t], &post);
    // At this point, post will contain log-likes of the selected
    // Gaussians.
    post.ApplySoftMax(); // Now they are posteriors (which sum to one).
    for (int32 i = 0; i < post.Dim(); i++) {
      // The next few lines (where we set up "input_chunk") are identical
      // to ApplyProjection.
      int32 gauss = gselect[t][i];
      SubVector<BaseFloat> this_stddev(stddevs_, gauss);
      BaseFloat this_post = post(i);
      input_chunk.Range(0, dim).AddVecVec(-this_post, gmm_.means_invvars().Row(gauss),
                                          this_stddev, 0.0);
      input_chunk.Range(0, dim).AddVecDivVec(this_post, this_feat, this_stddev,
                                             1.0);
      input_chunk(dim) = this_post;

      // If not for accumulating the + and - parts separately, we would be
      // doing something like:
      // proj_deriv_.Range(0, dim*ncontexts, gauss*(dim+1), dim+1).AddVecVec(
      //                    1.0, this_intermed_feat_deriv, input_chunk);


      SubMatrix<BaseFloat> plus_chunk(*proj_deriv_plus, 0, dim*ncontexts,
                                      gauss*(dim+1), dim+1),
          minus_chunk(*proj_deriv_minus, 0, dim*ncontexts,
                      gauss*(dim+1), dim+1);
          
      // This next function takes the rank-one matrix
      //  (this_intermed_deriv * input_chunk') and adds the positive
      // part to proj_deriv_plus, and minus the negative part to
      // proj_deriv_minus.
      AddOuterProductPlusMinus(static_cast<BaseFloat>(1.0),
                               this_intermed_feat_deriv,
                               input_chunk,
                               &plus_chunk, &minus_chunk);
    }
  }
}      

void Fmpe::ComputeFeatures(const MatrixBase<BaseFloat> &feat_in,
                           const std::vector<std::vector<int32> > &gselect,
                           Matrix<BaseFloat> *feat_out) const {
  int32 dim = FeatDim();
  KALDI_ASSERT(feat_in.NumRows() != 0 && feat_in.NumCols() == dim);
  KALDI_ASSERT(feat_in.NumRows() == static_cast<int32>(gselect.size()));
  feat_out->Resize(feat_in.NumRows(), feat_in.NumCols()); // will zero it.
  
  // Intermediate-dimension features
  Matrix<BaseFloat> intermed_feat(feat_in.NumRows(),
                                  dim * NumContexts());

  // Apply the main projection, from high-dim to intermediate
  // dimension (dim * NumContexts()).
  ApplyProjection(feat_in, gselect, &intermed_feat);

  // Apply the temporal context and reduces from
  // dimension dim*ncontexts to dim.
  ApplyContext(intermed_feat, feat_out);

  // Lastly, apply the the "C" matrix-- linear transform on the offsets.
  ApplyC(feat_out);
}


void Fmpe::AccStats(const MatrixBase<BaseFloat> &feat_in,
                    const std::vector<std::vector<int32> > &gselect,
                    const MatrixBase<BaseFloat> &feat_deriv_in,
                    MatrixBase<BaseFloat> *proj_deriv_plus,
                    MatrixBase<BaseFloat> *proj_deriv_minus) const {
  int32 dim = FeatDim(), ncontexts = NumContexts();
  KALDI_ASSERT(feat_in.NumRows() != 0 && feat_in.NumCols() == dim);
  KALDI_ASSERT(feat_in.NumRows() == static_cast<int32>(gselect.size()));
  AssertSameDim(*proj_deriv_plus, proj_);
  AssertSameDim(*proj_deriv_minus, proj_);
  AssertSameDim(feat_in, feat_deriv_in);

  // We do everything in reverse now, in reverse order.
  Matrix<BaseFloat> feat_deriv(feat_deriv_in);
  ApplyCReverse(&feat_deriv);

  Matrix<BaseFloat> intermed_feat_deriv(feat_in.NumRows(), dim*ncontexts);
  ApplyContextReverse(feat_deriv, &intermed_feat_deriv);
  
  ApplyProjectionReverse(feat_in, gselect, intermed_feat_deriv,
                         proj_deriv_plus, proj_deriv_minus);
}


void FmpeOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, context_expansion);
  WriteBasicType(os, binary, post_scale);
}
void FmpeOptions::Read(std::istream &is, bool binary) {
  ReadToken(is, binary, &context_expansion);
  ReadBasicType(is, binary, &post_scale);
}

Fmpe::Fmpe(const DiagGmm &gmm, const FmpeOptions &config): gmm_(gmm),
                                                          config_(config) {
  SetContexts(config.context_expansion);
  ComputeC();
  ComputeStddevs();
  proj_.Resize(NumGauss() * (FeatDim()+1), FeatDim() * NumContexts());
}

void Fmpe::Update(const FmpeUpdateOptions &config,
                  MatrixBase<BaseFloat> &proj_deriv_plus,
                  MatrixBase<BaseFloat> &proj_deriv_minus) {
  // tot_linear_objf_impr is the change in the actual
  // objective function if it were linear, i.e.
  //   objf-gradient . parameter-change  // Note: none of this is normalized by the #frames (we don't have
  // this info here), so that is done at the script level.
  BaseFloat tot_linear_objf_impr = 0.0;
  AssertSameDim(proj_deriv_plus, proj_);
  AssertSameDim(proj_deriv_minus, proj_);
  KALDI_ASSERT(proj_deriv_plus.Min() >= 0);
  KALDI_ASSERT(proj_deriv_minus.Min() >= 0);
  BaseFloat learning_rate = config.learning_rate,
      l2_weight = config.l2_weight;
  
  for (int32 i = 0; i < proj_.NumRows(); i++) {
    for (int32 j = 0; j < proj_.NumCols(); j++) {
      BaseFloat p = proj_deriv_plus(i,j), n = proj_deriv_minus(i,j),
          x = proj_(i,j);
      // Suppose the basic update (before regularization) is:
      // z <-- x  +   learning_rate * (p - n) / (p + n),
      // where z is the new parameter and x is the old one.
      // Here, we view (learning_rate / (p + n)) as a parameter-specific
      // learning rate.  In fact we view this update as the maximization
      // of an auxiliary function of the form:
      //  (z-x).(p-n)    - 0.5 (z - x)^2 (p+n)/learning_rate
      // and taking the derivative w.r.t z, we get:
      // Q'(z) =  (p-n) - (z - x) (p+n) / learning_rate
      // which we set to zero and solve for z, to get z = x + learning_rate.(p-n)/(p+n)
      // At this point we add regularization, a term of the form -l2_weight * z^2.
      // Our new auxiliary function derivative is:
      // Q(z) = -2.l2_weight.z + (p-n) - (z - x) (p+n) / learning_rate
      // We can write this as:
      // Q(z) = z . (-2.l2_weight - (p+n)/learning_rate)
      //        + (p-n) + x(p+n)/learning_rate
      // solving for z, we get:
      //      z = ((p-n) + x (p+n)/learning_rate) / (2.l2_weight + (p+n)/learning_rate)

      BaseFloat z = ((p-n) + x*(p+n)/learning_rate) / (2*l2_weight + (p+n)/learning_rate);
      // z is the new parameter value.

      tot_linear_objf_impr += (z-x) * (p-n); // objf impr based on linear assumption.
      proj_(i,j) = z;
    }
  }
  KALDI_LOG << "Objf impr (assuming linear) is " << tot_linear_objf_impr;
}

// Note: we write the GMM first, without any other header.
// This way, the gselect code can treat the form on disk as
// a normal GMM object.
void Fmpe::Write(std::ostream &os, bool binary) const {
  if (gmm_.NumGauss() == 0)
    KALDI_ERR << "Fmpe::Write, object not initialized.";
  gmm_.Write(os, binary);
  config_.Write(os, binary);
  // stddevs_ are derived, don't write them.
  proj_.Write(os, binary);
  C_.Write(os, binary);
  // contexts_ are derived from config, don't write them.
}


void Fmpe::Read(std::istream &is, bool binary) {
  gmm_.Read(is, binary);
  config_.Read(is, binary);
  ComputeStddevs(); // computed from gmm.
  proj_.Read(is, binary);
  C_.Read(is, binary);
  SetContexts(config_.context_expansion);
}



}  // End of namespace kaldi
