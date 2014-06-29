// transform/fmpe.cc

// Copyright 2011-2012  Yanmin Qian  Johns Hopkins University (Author: Daniel Povey)

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


#include "transform/fmpe.h"
#include "util/text-utils.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

namespace kaldi {

void Fmpe::SetContexts(std::string context_str) {
  // sets the contexts_ variable.
  using std::vector;
  using std::string;
  contexts_.clear();
  vector<string> ctx_vec; // splitting context_str on ":"
  SplitStringToVector(context_str, ":", false, &ctx_vec);
  contexts_.resize(ctx_vec.size());
  for (size_t i = 0; i < ctx_vec.size(); i++) {
    vector<string> pair_vec; // splitting ctx_vec[i] on ";"
    SplitStringToVector(ctx_vec[i], ";", false, &pair_vec);
    KALDI_ASSERT(pair_vec.size() != 0 && "empty context!");
    for (size_t j = 0; j < pair_vec.size(); j++) {
      vector<string> one_pair;
      SplitStringToVector(pair_vec[j], ",", false, &one_pair);
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
    x2_stats.AddDiagVec(ngmm.weights_(pdf), ngmm.vars_.Row(pdf)); // add diagonal
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
    TpMatrix<double> Ctmp(dim); Ctmp.Cholesky(x2_stats);
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
  KALDI_ASSERT(intermed_feat.NumCols() == dim * ncontexts &&
               intermed_feat.NumRows() == feat_out->NumRows()
               && feat_out->NumCols() == dim);
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
  KALDI_ASSERT(intermed_feat_deriv->NumCols() == dim * ncontexts &&
               intermed_feat_deriv->NumRows() == feat_deriv.NumRows()
               && feat_deriv.NumCols() == dim);
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
        int32 t_in = t_out + t_offset; // t_in indexes the input.
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

// Constructs the high-dim features and applies the main projection matrix
// projT_.  This projects from dimension ngauss*(dim+1) to dim*ncontexts.  Note:
// because the input vector of size ngauss*(dim+1) is sparse in a blocky way
// (i.e. each frame only has a couple of nonzero posteriors), we deal with
// sub-matrices of the projection matrix projT_.  We actually further optimize
// the code by taking all frames in a file that had nonzero posteriors for a
// particular Gaussian, and forming a matrix out of the corresponding
// high-dimensional features; we can then use a matrix-matrix multiply rather
// than using vector-matrix operations.

void Fmpe::ApplyProjection(const MatrixBase<BaseFloat> &feat_in,
                           const std::vector<std::vector<int32> > &gselect,
                           MatrixBase<BaseFloat> *intermed_feat) const {
  int32 dim = FeatDim(), ncontexts = NumContexts();  
  
  Vector<BaseFloat> post; // will be posteriors of selected Gaussians.
  Vector<BaseFloat> input_chunk(dim+1); // will be a segment of
  // the high-dimensional features.

  // "all_posts" is a vector of ((gauss-index, time-index), gaussian
  // posterior).
  // We'll compute the posterior information, sort it, and then
  // go through it in sorted order, which maintains memory locality
  // when accessing the projection matrix.
  // Note: if we really cared we could make this use level-3 BLAS
  // (matrix-matrix multiply), but we'd need to have a temporary
  // matrix for the output and input.
  std::vector<std::pair<std::pair<int32, int32>, BaseFloat> > all_posts;
  
  for (int32 t = 0; t < feat_in.NumRows(); t++) {
    SubVector<BaseFloat> this_feat(feat_in, t);
    gmm_.LogLikelihoodsPreselect(this_feat, gselect[t], &post);
    // At this point, post will contain log-likes of the selected
    // Gaussians.
    post.ApplySoftMax(); // Now they are posteriors (which sum to one).
    for (int32 i = 0; i < post.Dim(); i++) {
      int32 gauss = gselect[t][i];
      all_posts.push_back(std::make_pair(std::make_pair(gauss, t), post(i)));
    }
  }
  std::sort(all_posts.begin(), all_posts.end());
  
  bool optimize = true;

  if (!optimize) { // Why do we keep this un-optimized code around?
    // For clarity, so you can see what's going on, and for easier
    // comparision with ApplyProjectionReverse which is similar to this
    // un-optimized segment.  Both un-optimized and optimized versions
    // should give identical transforms (up to tiny roundoff differences).
    for (size_t i = 0; i < all_posts.size(); i++) {
      int32 gauss = all_posts[i].first.first, t = all_posts[i].first.second;
      SubVector<BaseFloat> this_feat(feat_in, t);
      SubVector<BaseFloat> this_intermed_feat(*intermed_feat, t);
      BaseFloat this_post = all_posts[i].second;
      SubVector<BaseFloat> this_stddev(stddevs_, gauss);

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
      input_chunk(dim) = this_post * config_.post_scale;

      // this_intermed_feat += [appropriate chjunk of projT_] * input_chunk.
      this_intermed_feat.AddMatVec(1.0, projT_.Range(gauss*(dim+1), dim+1,
                                                     0, dim*ncontexts),
                                   kTrans, input_chunk, 1.0);
    }
  } else {
    size_t i = 0;
    // We process the "posts" vector in chunks, where each chunk corresponds to
    // the same Gaussian index (but different times).
    while (i < all_posts.size()) {
      int32 gauss = all_posts[i].first.first;
      SubVector<BaseFloat> this_stddev(stddevs_, gauss),
          this_mean_invvar(gmm_.means_invvars(), gauss);
      SubMatrix<BaseFloat> this_projT_chunk(projT_, gauss*(dim+1), dim+1,
                                            0, dim*ncontexts);
      int32 batch_size; // number of posteriors with same Gaussian..
      for (batch_size = 0;
           batch_size+i < static_cast<int32>(all_posts.size()) &&
               all_posts[batch_size+i].first.first == gauss;
           batch_size++); // empty loop body.
      Matrix<BaseFloat> input_chunks(batch_size, dim+1);
      Matrix<BaseFloat> intermed_temp(batch_size, dim*ncontexts);
      for (int32 j = 0; j < batch_size; j++) { // set up "input_chunks".
        // To understand this code, first examine code and comments in "non-optimized"
        // code chunk above (the other branch of the if/else statement).
        int32 t = all_posts[i+j].first.second;
        SubVector<BaseFloat> this_feat(feat_in, t);
        SubVector<BaseFloat> this_input_chunk(input_chunks, j);
        BaseFloat this_post = all_posts[i+j].second;
        this_input_chunk.Range(0, dim).AddVecVec(-this_post,
                                                 this_mean_invvar,
                                                 this_stddev, 0.0);
        this_input_chunk.Range(0, dim).AddVecDivVec(this_post, this_feat,
                                                    this_stddev, 1.0);
        this_input_chunk(dim) = this_post * config_.post_scale;
      }
      // The next line is where most of the computation will happen,
      // during the feature computation phase.  We have rearranged
      // stuff so it's a matrix-matrix operation, for greater
      // efficiency (when using optimized libraries like ATLAS).
      intermed_temp.AddMatMat(1.0, input_chunks, kNoTrans,
                              this_projT_chunk, kNoTrans, 0.0);
      for (int32 j = 0; j < batch_size; j++) { // add data from
        // intermed_temp to the output "intermed_feat"
        int32 t = all_posts[i+j].first.second;
        SubVector<BaseFloat> this_intermed_feat(*intermed_feat, t);
        SubVector<BaseFloat> this_intermed_temp(intermed_temp, j);
        // this_intermed_feat += this_intermed_temp.
        this_intermed_feat.AddVec(1.0, this_intermed_temp);
      }
      i += batch_size;
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

  // "all_posts" is a vector of ((gauss-index, time-index), gaussian
  // posterior).
  // We'll compute the posterior information, sort it, and then
  // go through it in sorted order, which maintains memory locality
  // when accessing the projection matrix.
  std::vector<std::pair<std::pair<int32, int32>, BaseFloat> > all_posts;
  
  for (int32 t = 0; t < feat_in.NumRows(); t++) {
    SubVector<BaseFloat> this_feat(feat_in, t);
    gmm_.LogLikelihoodsPreselect(this_feat, gselect[t], &post);
    // At this point, post will contain log-likes of the selected
    // Gaussians.
    post.ApplySoftMax(); // Now they are posteriors (which sum to one).
    for (int32 i = 0; i < post.Dim(); i++) {
      // The next few lines (where we set up "input_chunk") are identical
      // to ApplyProjection.
      int32 gauss = gselect[t][i];
      all_posts.push_back(std::make_pair(std::make_pair(gauss, t), post(i)));
    }
  }
  std::sort(all_posts.begin(), all_posts.end());
  for (size_t i = 0; i < all_posts.size(); i++) {
    int32 gauss = all_posts[i].first.first, t = all_posts[i].first.second;
    BaseFloat this_post = all_posts[i].second;
    SubVector<BaseFloat> this_feat(feat_in, t);    
    SubVector<BaseFloat> this_intermed_feat_deriv(intermed_feat_deriv, t);
    SubVector<BaseFloat> this_stddev(stddevs_, gauss);
    input_chunk.Range(0, dim).AddVecVec(-this_post, gmm_.means_invvars().Row(gauss),
                                        this_stddev, 0.0);
    input_chunk.Range(0, dim).AddVecDivVec(this_post, this_feat, this_stddev,
                                           1.0);
    input_chunk(dim) = this_post * config_.post_scale;

    // If not for accumulating the + and - parts separately, we would be
    // doing something like:
    // proj_deriv_.Range(0, dim*ncontexts, gauss*(dim+1), dim+1).AddVecVec(
    //                    1.0, this_intermed_feat_deriv, input_chunk);


    SubMatrix<BaseFloat> plus_chunk(*proj_deriv_plus, 
                                    gauss*(dim+1), dim+1,
                                    0, dim*ncontexts),
        minus_chunk(*proj_deriv_minus, 
                    gauss*(dim+1), dim+1,
                    0, dim*ncontexts);
          
    // This next function takes the rank-one matrix
    //  (input_chunk * this_intermed_deriv'), and adds the positive
    // part to proj_deriv_plus, and minus the negative part to
    // proj_deriv_minus.
    AddOuterProductPlusMinus(static_cast<BaseFloat>(1.0),
                             input_chunk,
                             this_intermed_feat_deriv,
                             &plus_chunk, &minus_chunk);
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
                    const MatrixBase<BaseFloat> &direct_feat_deriv,
                    const MatrixBase<BaseFloat> *indirect_feat_deriv, // may be NULL
                    FmpeStats *fmpe_stats) const {
  SubMatrix<BaseFloat> stats_plus(fmpe_stats->DerivPlus());
  SubMatrix<BaseFloat> stats_minus(fmpe_stats->DerivMinus());
  int32 dim = FeatDim(), ncontexts = NumContexts();
  KALDI_ASSERT(feat_in.NumRows() != 0 && feat_in.NumCols() == dim);
  KALDI_ASSERT(feat_in.NumRows() == static_cast<int32>(gselect.size()));
  KALDI_ASSERT(SameDim(stats_plus, projT_) && SameDim(stats_minus, projT_) &&
               SameDim(feat_in, direct_feat_deriv));

  if (indirect_feat_deriv != NULL)
    fmpe_stats->AccumulateChecks(feat_in, direct_feat_deriv, *indirect_feat_deriv);
  
  Matrix<BaseFloat> feat_deriv(direct_feat_deriv); // "feat_deriv" is initially direct+indirect.
  if (indirect_feat_deriv != NULL)
    feat_deriv.AddMat(1.0, *indirect_feat_deriv);
  
  // We do the "*Reverse" version of each stage now, in reverse order.
  ApplyCReverse(&feat_deriv);
  
  Matrix<BaseFloat> intermed_feat_deriv(feat_in.NumRows(), dim*ncontexts);
  ApplyContextReverse(feat_deriv, &intermed_feat_deriv);
  
  ApplyProjectionReverse(feat_in, gselect, intermed_feat_deriv,
                         &stats_plus, &stats_minus);
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
  projT_.Resize(NumGauss() * (FeatDim()+1), FeatDim() * NumContexts());
}

BaseFloat Fmpe::Update(const FmpeUpdateOptions &config,
                       const FmpeStats &stats) {
  SubMatrix<BaseFloat> proj_deriv_plus = stats.DerivPlus(),
      proj_deriv_minus = stats.DerivMinus();
  // tot_linear_objf_impr is the change in the actual
  // objective function if it were linear, i.e.
  //   objf-gradient . parameter-change
  // Note: none of this is normalized by the #frames (we don't have
  // this info here), so that is done at the script level.
  BaseFloat tot_linear_objf_impr = 0.0;
  int32 changed = 0; // Keep track of how many elements change sign.
  KALDI_ASSERT(SameDim(proj_deriv_plus, projT_) && SameDim(proj_deriv_minus, projT_));
  KALDI_ASSERT(proj_deriv_plus.Min() >= 0);
  KALDI_ASSERT(proj_deriv_minus.Min() >= 0);
  BaseFloat learning_rate = config.learning_rate,
      l2_weight = config.l2_weight;
  
  for (int32 i = 0; i < projT_.NumRows(); i++) {
    for (int32 j = 0; j < projT_.NumCols(); j++) {
      BaseFloat p = proj_deriv_plus(i, j), n = proj_deriv_minus(i, j),
          x = projT_(i, j);
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
      projT_(i, j) = z;
      if (z*x < 0) changed++;
    }
  }
  KALDI_LOG << "Objf impr (assuming linear) is " << tot_linear_objf_impr;
  KALDI_LOG << ((100.0*changed)/(projT_.NumRows()*projT_.NumCols()))
            << "% of matrix elements changed sign.";
  return tot_linear_objf_impr;
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
  projT_.Write(os, binary);
  C_.Write(os, binary);
  // contexts_ are derived from config, don't write them.
}


void Fmpe::Read(std::istream &is, bool binary) {
  gmm_.Read(is, binary);
  config_.Read(is, binary);
  ComputeStddevs(); // computed from gmm.
  projT_.Read(is, binary);
  C_.Read(is, binary);
  SetContexts(config_.context_expansion);
}


BaseFloat ComputeAmGmmFeatureDeriv(const AmDiagGmm &am_gmm,
                                   const TransitionModel &trans_model,
                                   const Posterior &posterior,
                                   const MatrixBase<BaseFloat> &features,
                                   Matrix<BaseFloat> *direct_deriv,
                                   const AccumAmDiagGmm *model_diff,
                                   Matrix<BaseFloat> *indirect_deriv) {
  KALDI_ASSERT((model_diff != NULL) == (indirect_deriv != NULL));
  BaseFloat ans = 0.0;
  KALDI_ASSERT(posterior.size() == static_cast<size_t>(features.NumRows()));
  direct_deriv->Resize(features.NumRows(), features.NumCols());
  if (indirect_deriv != NULL)
    indirect_deriv->Resize(features.NumRows(), features.NumCols());
  Vector<BaseFloat> temp_vec(features.NumCols());
  Vector<double> temp_vec_dbl(features.NumCols());
  for (size_t i = 0; i < posterior.size(); i++) {
    for (size_t j = 0; j < posterior[i].size(); j++) {
      int32 tid = posterior[i][j].first,  // transition identifier.
          pdf_id = trans_model.TransitionIdToPdf(tid);
      BaseFloat weight = posterior[i][j].second;
      const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
      Vector<BaseFloat> gauss_posteriors;
      SubVector<BaseFloat> this_feat(features, i);
      SubVector<BaseFloat> this_direct_deriv(*direct_deriv, i);
      ans += weight * 
          gmm.ComponentPosteriors(this_feat, &gauss_posteriors);
      
      gauss_posteriors.Scale(weight);
      // The next line does: to i'th row of deriv, add
      // means_invvars^T * gauss_posteriors,
      // where each row of means_invvars is the mean times
      // diagonal inverse covariance... after transposing,
      // this becomes a weighted of these rows, weighted by
      // the posteriors.  This comes from the term
      //  feat^T * inv_var * mean
      // in the objective function.
      this_direct_deriv.AddMatVec(1.0, gmm.means_invvars(), kTrans,
                                  gauss_posteriors, 1.0);      

      // next line does temp_vec == inv_vars^T * gauss_posteriors,
      // which sets temp_vec to a weighted sum of the inv_vars,
      // weighed by Gaussian posterior.
      temp_vec.AddMatVec(1.0, gmm.inv_vars(), kTrans,
                         gauss_posteriors, 0.0);
      // Add to the derivative, -(this_feat .* temp_vec),
      // which is the term that comes from the -0.5 * inv_var^T feat_sq,
      // in the objective function (where inv_var is a vector, and feat_sq
      // is a vector of squares of the feature values).
      // Note: we have to do some messing about with double-precision here
      // because the stats only come in double precision.
      this_direct_deriv.AddVecVec(-1.0, this_feat, temp_vec, 1.0);
      if (model_diff != NULL && weight > 0.0) { // We need to get the indirect diff.
        // This "weight > 0.0" checks that this is the numerator stats, as the
        // fMPE indirect diff applies only to the ML stats-- CAUTION, this
        // code will only work as-is for fMMI (and the stats should not be
        // canceled), due to the assumption that ML stats == num stats.
        Vector<double> gauss_posteriors_dbl(gauss_posteriors);
        const AccumDiagGmm &deriv_acc = model_diff->GetAcc(pdf_id);
        // part of the derivative.  Note: we could just store the direct and
        // indirect derivatives together in one matrix, but it makes it easier
        // to accumulate certain diagnostics if we store them separately.
        SubVector<BaseFloat> this_indirect_deriv(*indirect_deriv, i);
        // note: deriv_acc.mean_accumulator() contains the derivative of
        // the objective function w.r.t. the "x stats" accumulated for
        // this GMM.  variance_accumulator() is the same for the "x^2 stats".
        temp_vec_dbl.AddMatVec(1.0, deriv_acc.mean_accumulator(), kTrans,
                               gauss_posteriors_dbl, 0.0);
        this_indirect_deriv.AddVec(1.0, temp_vec_dbl);
        temp_vec_dbl.AddMatVec(1.0, deriv_acc.variance_accumulator(), kTrans,
                               gauss_posteriors_dbl, 0.0);
        temp_vec.CopyFromVec(temp_vec_dbl); // convert to float.
        // next line because d(x^2 stats for Gaussian)/d(feature) =
        // 2 * (gaussian posterior) * feature.
        this_indirect_deriv.AddVecVec(2.0, this_feat, temp_vec, 1.0);
      }
    }
  }
  return ans;
}


SubMatrix<BaseFloat> FmpeStats::DerivPlus() const { // const-ness not preserved.
  KALDI_ASSERT(deriv.NumRows() != 0);
  int32 proj_num_rows = deriv.NumRows(),
      proj_num_cols = deriv.NumCols()/2;
  return SubMatrix<BaseFloat>(deriv, 0, proj_num_rows,
                              0, proj_num_cols);
}
SubMatrix<BaseFloat> FmpeStats::DerivMinus() const { // const-ness not preserved.
  KALDI_ASSERT(deriv.NumRows() != 0);
  int32 proj_num_rows = deriv.NumRows(),
      proj_num_cols = deriv.NumCols()/2;
  return SubMatrix<BaseFloat>(deriv, 0, proj_num_rows,
                              proj_num_cols, proj_num_cols);
}

void FmpeStats::Init(const Fmpe &fmpe) {
  int32 num_rows = fmpe.ProjectionTNumRows(),
      num_cols = fmpe.ProjectionTNumCols();
  deriv.Resize(num_rows, num_cols*2);

  int32 feat_dim = fmpe.FeatDim();
  checks.Resize(8, feat_dim);
}

void FmpeStats::AccumulateChecks(const MatrixBase<BaseFloat> &feats,
                                 const MatrixBase<BaseFloat> &direct_deriv,
                                 const MatrixBase<BaseFloat> &indirect_deriv) {
  int32 T = feats.NumRows(), dim = feats.NumCols();
  KALDI_ASSERT(direct_deriv.NumRows() == T && direct_deriv.NumCols() == dim &&
               indirect_deriv.NumRows() == T && indirect_deriv.NumCols() == dim);
  KALDI_ASSERT(checks.NumRows() == 8 && checks.NumCols() == dim);
  for (int32 t = 0; t < T; t++) {
    for (int32 d = 0; d < dim; d++) {
      BaseFloat zero = 0.0;
      checks(0, d) += std::max(zero, direct_deriv(t, d));
      checks(1, d) += std::max(zero, -direct_deriv(t, d));
      checks(2, d) += std::max(zero, indirect_deriv(t, d));
      checks(3, d) += std::max(zero, -indirect_deriv(t, d));
      checks(4, d) += std::max(zero, feats(t, d)*direct_deriv(t, d));
      checks(5, d) += std::max(zero, -feats(t, d)*direct_deriv(t, d));
      checks(6, d) += std::max(zero, feats(t, d)*indirect_deriv(t, d));
      checks(7, d) += std::max(zero, -feats(t, d)*indirect_deriv(t, d));
    }
  }
}

void FmpeStats::DoChecks() {
  if (checks.IsZero()) {
    KALDI_LOG << "No checks will be done, probably indirect derivative was not used.";
    return;
  }
  int32 dim = checks.NumCols();
  Vector<double> shift_check(dim), shift_check2(dim), scale_check(dim), scale_check2(dim);
  for (int32 d = 0; d < dim; d++) {
    // shiftnumerator = direct+indirect deriv-- should be zero.
    double shift_num = checks(0, d) - checks(1, d) + checks(2, d) - checks(3, d),
        shift_den = checks(0, d) + checks(1, d) + checks(2, d) + checks(3, d),
        shift_den2 = fabs(checks(0, d) - checks(1, d)) + fabs(checks(2, d) - checks(3, d));
    shift_check(d) = shift_num / shift_den;
    shift_check2(d) = shift_num / shift_den2;
    double scale_num = checks(4, d) - checks(5, d) + checks(6, d) - checks(7, d),
        scale_den = checks(4, d) + checks(5, d) + checks(6, d) + checks(7, d),
        scale_den2 = fabs(checks(4, d) - checks(5, d)) + fabs(checks(6, d) - checks(7, d));
    scale_check(d) = scale_num / scale_den;
    scale_check2(d) = scale_num / scale_den2;
  }

  KALDI_LOG << "Shift-check is as follows (should be in range +- 0.01 or less)."
            << shift_check;
  KALDI_LOG << "Scale-check is as follows (should be in range +- 0.01 or less)."
            << scale_check;
  KALDI_LOG << "Shift-check(2) is as follows: most elements should be in range +-0.1: "
            << shift_check2;
  KALDI_LOG << "Scale-check(2) is as follows: most elements should be in range +-0.1: "
            << scale_check2;
}

void FmpeStats::Write(std::ostream &os, bool binary) const {
  deriv.Write(os, binary);
  checks.Write(os, binary);
}

void FmpeStats::Read(std::istream &is, bool binary, bool add) {
  deriv.Read(is, binary, add);
  checks.Read(is, binary, add);
}


}  // End of namespace kaldi
