// transform/fmllr-raw.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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

#include <utility>
#include <vector>
using std::vector;

#include "transform/fmllr-raw.h"
#include "transform/fmllr-diag-gmm.h"

namespace kaldi {

FmllrRawAccs::FmllrRawAccs(int32 raw_dim,
                           int32 model_dim,
                           const Matrix<BaseFloat> &full_transform):
    raw_dim_(raw_dim),
    model_dim_(model_dim) {
  if (full_transform.NumCols() != full_transform.NumRows() &&
      full_transform.NumCols() != full_transform.NumRows() + 1) {
    KALDI_ERR << "Expecting full LDA+MLLT transform to be square or d by d+1 "
              << "(make sure you are including rejected rows).";
  }
  if (raw_dim <= 0 || full_transform.NumRows() % raw_dim != 0)
    KALDI_ERR << "Raw feature dimension is invalid " << raw_dim
              << "(must be positive and divide feature dimension)";
  int32 full_dim = full_transform.NumRows();
  full_transform_ = full_transform.Range(0, full_dim, 0, full_dim);
  transform_offset_.Resize(full_dim);
  if (full_transform_.NumCols() == full_dim + 1)
    transform_offset_.CopyColFromMat(full_transform_, full_dim);
  
  int32 full_dim2 = ((full_dim+1)*(full_dim+2))/2;
  count_ = 0.0;

  temp_.Resize(full_dim + 1);
  Q_.Resize(model_dim + 1, full_dim + 1);
  S_.Resize(model_dim + 1, full_dim2);

  single_frame_stats_.s.Resize(full_dim + 1);
  single_frame_stats_.transformed_data.Resize(full_dim);
  single_frame_stats_.count = 0.0;
  single_frame_stats_.a.Resize(model_dim);
  single_frame_stats_.b.Resize(model_dim);
}


bool FmllrRawAccs::DataHasChanged(const VectorBase<BaseFloat> &data) const {
  KALDI_ASSERT(data.Dim() == FullDim());
  return !data.ApproxEqual(single_frame_stats_.s.Range(0, FullDim()), 0.0);
}

void FmllrRawAccs::CommitSingleFrameStats() {
  // Commit the stats for this from (in SingleFrameStats).
  int32 model_dim = ModelDim(), full_dim = FullDim();
  SingleFrameStats &stats = single_frame_stats_;
  if (stats.count == 0.0) return;

  count_ += stats.count;

  // a_ext and b_ext are a and b extended with the count,
  // which we'll later use to reconstruct the full stats for
  // the rejected dimensions.
  Vector<double> a_ext(model_dim + 1), b_ext(model_dim + 1);
  a_ext.Range(0, model_dim).CopyFromVec(stats.a);
  b_ext.Range(0, model_dim).CopyFromVec(stats.b);
  a_ext(model_dim) = stats.count;
  b_ext(model_dim) = stats.count;
  Q_.AddVecVec(1.0, a_ext, Vector<double>(stats.s));

  temp_.SetZero();
  temp_.AddVec2(1.0, stats.s);
  int32 full_dim2 = ((full_dim + 1) * (full_dim + 2)) / 2;
  SubVector<double> temp_vec(temp_.Data(), full_dim2);
  S_.AddVecVec(1.0, b_ext, temp_vec);
}

void FmllrRawAccs::InitSingleFrameStats(const VectorBase<BaseFloat> &data) {
  SingleFrameStats &stats = single_frame_stats_;
  int32 full_dim = FullDim();
  KALDI_ASSERT(data.Dim() == full_dim);
  stats.s.Range(0, full_dim).CopyFromVec(data);
  stats.s(full_dim) = 1.0;
  stats.transformed_data.AddMatVec(1.0, full_transform_, kNoTrans, data, 0.0);
  stats.transformed_data.AddVec(1.0, transform_offset_);
  stats.count = 0.0;
  stats.a.SetZero();
  stats.b.SetZero();
}


BaseFloat FmllrRawAccs::AccumulateForGmm(const DiagGmm &gmm,
                                         const VectorBase<BaseFloat> &data,
                                         BaseFloat weight) {
  int32 model_dim = ModelDim(), full_dim = FullDim();
  KALDI_ASSERT(data.Dim() == full_dim &&
               "Expect raw, spliced data, which should have same dimension as "
               "full transform.");
  if (DataHasChanged(data)) {
    // this is part of our mechanism to accumulate certain sub-parts of
    // the computation for each frame, to avoid excessive compute.
    CommitSingleFrameStats();
    InitSingleFrameStats(data);
  }
  SingleFrameStats &stats = single_frame_stats_;

  SubVector<BaseFloat> projected_data(stats.transformed_data, 0, model_dim);

  int32 num_gauss = gmm.NumGauss();
  Vector<BaseFloat> posterior(num_gauss);
  BaseFloat log_like = gmm.ComponentPosteriors(projected_data, &posterior);
  posterior.Scale(weight);
  // Note: AccumulateFromPosteriors takes the original, spliced data,
  // and returns the log-like of the rejected dimensions.
  AccumulateFromPosteriors(gmm, data, posterior);

  // Add the likelihood of the rejected dimensions to the objective function
  // (assume zero-mean, unit-variance Gaussian; the LDA should have any offset
  // required to ensure this).
  if (full_dim > model_dim) {
    SubVector<BaseFloat> rejected_data(stats.transformed_data,
                                       model_dim, full_dim - model_dim);
    log_like += -0.5 * (VecVec(rejected_data, rejected_data)
                        + (full_dim - model_dim) * M_LOG_2PI);
  }
  return log_like;
}

/*
  // Extended comment here.
  //
  // Let x_t(i) be the fully processed feature, dimension i (with fMLLR transform
  //  and LDA transform), but *without* any offset term from the LDA, which
  //  it's more convenient to view as an offset in the model.
  //
  //
  // For a given dimension i (either accepted or rejected), the auxf can
  // be expressed as a quadratic function of x_t(i).  We ultimately will want to
  // express x_t(i) as a linear function of the parameters of the linearized
  // fMLLR transform matrix.  Some notation:
  //    Let l be the linearized transform matrix, i.e. the concatenation of the
  //       m rows, each of length m+1, of the fMLLR transform.
  //    Let n be the number of frames we splice together each time.
  //    Let s_t be the spliced-together features on time t, with a one appended;
  //       it will have n blocks each of size m, followed by a 1.  (dim is n*m + 1).
  //     
  // x(i) [note, this is the feature without any LDA offset], is bilinear in the
  //      transform matrix and the features, so:
  //
  // x(i) = l^T M_i s_t, where s_t is the spliced features on time t,
  //          with a 1 appended
  //   [we need to compute M_i but we know the function is bilinear so it exists].
  //
  // The auxf can be written as:
  // F = sum_i sum_t  a_{ti} x(i) - 0.5  b_{ti} x(i)^2 
  //   = sum_i sum_t  a_{ti} x(i) - 0.5  b_{ti} x(i)^2
  //   = sum_i sum_t  a_{ti} (l^T M_i s_t)  -  0.5 b_{ti} (l^T M_i s_t )^2
  //   = sum_i l^T M_i q_i  +  l^T M_i S_i M_i^T l 
  //  where
  //     q_i = sum_t a_{ti} s_t, and
  //     S_i = sum_t b_{ti} s_t s_t^T
  //   [Note that we only need store S_i for the model-dim plus one, because
  //    all the rejected dimensions have the same value]
  //
  //     We define a matrix Q whose rows are q_d, with
  //       Q = \sum_t d_t s_t^T
  //    [The Q we actually store as stats will use a modified form of d that
  //     has a 1 for all dimensions past the model dim, to avoid redundancy;
  //     we'll reconstruct the true Q from this later on.]
  //     
  //
  // What is M_i?  Working it out is a little tedious.
  //  Note: each M_i (from i = 0 ... full_dim) is of
  //    dimension (raw_dim*(raw_dim+1)) by full_dim + 1
  // 
  // We want to express x(i) [we forget the subscript "t" sometimes],
  // as a bilinear function of l and s_t.
  //    We have x(i) = l^T M_i s.
  //
  // The (j,k)'th component of M_i is the term in x(i) that corresponds to the j'th
  // component of l and the k'th of s.

  // Before defining M_i, let us define N_i, where l^t N_i s will equal the spliced and
  // transformed pre-LDA features of dimension i.  the N's have the same dimensions as the
  // M's.
  //
  // We'll first define the j,k'th component of N_i, as this is easier; we'll then define the M_i
  // as combinations of N_i.
  //
  // For a given i, j and k, the value of n_{i,j,k} will be as follows:
  //   We first decompose index j into j1, j2 (both functions of
  //    the original index j), where
  //    j1 corresponds to the row-index of the fMLLR transform, j2 to the col-index.
  //   We next decompose i into i1, i2, where i1 corresponds to the splicing number
  //   (0...n-1), and i2 corresponds to the cepstral index.
  //
  //   If (j1 != i2) then n_{ijk} == 0.
  //
  //   Elsif k corresponds to the last element [i.e. k == m * n], then this m_{ijk} corresponds
  //   to the effect of the j'th component of l for zero input, so:
  //     If j2 == m (i.e. this the offset term in the fMLLR matrix), then
  //       n_{ijk} = 1.0,
  //     Else
  //       n_{ijk} = 0.0
  //     Fi
  //
  //   Else:
  //     Decompose k into k1, k2, where k1 = 0.. n-1 is the splicing index, and k2 = 0...m-1 is
  //      the cepstral index.
  //     If k1 != i1 then
  //       n_{ijk} = 0.0
  //     elsif k2 != j2 then
  //       n_{ijk} = 0.0
  //     else
  //       n_{ijk} = 1.0
  //     fi
  //    Endif
  //    Now,  M_i will be defined as sum_i T_{ij} N_j, where T_{ij} are the elements of the
  //     LDA+MLLT transform (but excluding any linear offset, which gets accounted for by
  //     c_i, above).
  //
  //  Now suppose we want to express the auxiliary function in a simpler form
  //  as l^T v - 0.5 l^T W l, where v and W are the "simple" linear and quadratic stats,
  //  we can do so with:
  //     v = \sum_i M_i q_i   
  //  and
  //     W = \sum_i M_i S_i M_i^T
  //
  */

void FmllrRawAccs::AccumulateFromPosteriors(
    const DiagGmm &diag_gmm,
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posterior) {
  // The user may call this function directly, even though we also
  // call it from AccumulateForGmm(), so check again:
  if (DataHasChanged(data)) { 
    CommitSingleFrameStats();
    InitSingleFrameStats(data);
  }
  
  int32  model_dim = ModelDim();

  SingleFrameStats &stats = single_frame_stats_;
  
  // The quantities a and b describe the diagonal auxiliary function
  // for each of the retained dimensions in the transformed space--
  // in the format F = \sum_d alpha(d) x(d)  -0.5 beta(d) x(d)^2,
  // where x(d) is the d'th dimensional fully processed feature.
  // For d, see the comment-- it's alpha processed to take into
  // account any offset in the LDA.  Note that it's a reference.
  //
  Vector<double> a(model_dim), b(model_dim);
  
  int32 num_comp = diag_gmm.NumGauss();
  
  double count = 0.0; // data-count contribution from this frame.

  // Note: we could do this using matrix-matrix operations instead of
  // row by row.  In the end it won't really matter as this is not
  // the slowest part of the computation.
  for (size_t m = 0; m < num_comp; m++) {
    BaseFloat this_post = posterior(m);
    if (this_post != 0.0) {
      count += this_post;
      a.AddVec(this_post, diag_gmm.means_invvars().Row(m));
      b.AddVec(this_post, diag_gmm.inv_vars().Row(m));
    }
  }
  // Correct "a" for any offset term in the LDA transform-- we view it as
  // the opposite offset in the model [note: we'll handle the rejected dimensions
  // in update time.]  Here, multiplying the element of "b" (which is the
  // weighted inv-vars) by transform_offset_, and subtracting the result from
  // a, is like subtracting the transform-offset from the original means
  // (because a contains the means times inv-vars_.
  Vector<double> offset(transform_offset_.Range(0, model_dim));
  a.AddVecVec(-1.0, b, offset, 1.0);
  stats.a.AddVec(1.0, a);
  stats.b.AddVec(1.0, b);
  stats.count += count;
}


void FmllrRawAccs::Update(const FmllrRawOptions &opts,
                          MatrixBase<BaseFloat> *raw_fmllr_mat,
                          BaseFloat *objf_impr,
                          BaseFloat *count) {
  // First commit any pending stats from the last frame.
  if (single_frame_stats_.count != 0.0)
    CommitSingleFrameStats();
  
  if (this->count_ < opts.min_count) {
    KALDI_WARN << "Not updating (raw) fMLLR since count " << this->count_
               << " is less than min count " << opts.min_count;
    *objf_impr = 0.0;
    *count = this->count_;
    return;
  }
  KALDI_ASSERT(raw_fmllr_mat->NumRows() == RawDim() &&
               raw_fmllr_mat->NumCols() == RawDim() + 1 &&
               !raw_fmllr_mat->IsZero());
  Matrix<double> fmllr_mat(*raw_fmllr_mat); // temporary, double-precision version
                                            // of matrix.


  Matrix<double> linear_stats; // like K in diagonal update.
  std::vector<SpMatrix<double> > diag_stats; // like G in diagonal update.
                                             // Note: we will invert these.
  std::vector<std::vector<Matrix<double> > > off_diag_stats; // these will
  // contribute to the linear term.

  Vector<double> simple_linear_stats;
  SpMatrix<double> simple_quadratic_stats;
  ConvertToSimpleStats(&simple_linear_stats, &simple_quadratic_stats);
  
  ConvertToPerRowStats(simple_linear_stats, simple_quadratic_stats,
                       &linear_stats, &diag_stats, &off_diag_stats);

  try {
    for (size_t i = 0; i < diag_stats.size(); i++) {
      diag_stats[i].Invert();
    }
  } catch (...) {
    KALDI_WARN << "Error inverting stats matrices for fMLLR "
               << "[min-count too small?  Bad data?], not updating.";
    return;
  }
  
  int32 raw_dim = RawDim(), splice_width = SpliceWidth();
  
  double effective_beta = count_ * splice_width; // We "count" the determinant
  // splice_width times in the objective function.

  double auxf_orig = GetAuxf(simple_linear_stats, simple_quadratic_stats,
                             fmllr_mat);
  for (int32 iter = 0; iter < opts.num_iters; iter++) {
    for (int32 row = 0; row < raw_dim; row++) {
      SubVector<double> this_row(fmllr_mat, row);
      Vector<double> this_linear(raw_dim + 1);  // Here, k_i is the linear term
      // in the auxf expressed as a function of this row.
      this_linear.CopyFromVec(linear_stats.Row(row));
      for (int32 row2 = 0; row2 < raw_dim; row2++) {
        if (row2 != row) {
          if (row2 < row) {
            this_linear.AddMatVec(-1.0, off_diag_stats[row][row2], kNoTrans,
                                  fmllr_mat.Row(row2), 1.0);
          } else {
            // We won't have the element [row][row2] stored, but use symmetry.
            this_linear.AddMatVec(-1.0, off_diag_stats[row2][row], kTrans,
                                  fmllr_mat.Row(row2), 1.0);
          }
        }
      }
      FmllrInnerUpdate(diag_stats[row],
                       this_linear,
                       effective_beta,
                       row,
                       &fmllr_mat);
    }
  }
  double auxf_final = GetAuxf(simple_linear_stats, simple_quadratic_stats,
                              fmllr_mat),
      auxf_change = auxf_final - auxf_orig;
  *count = this->count_;
  KALDI_VLOG(1) << "Updating raw fMLLR: objf improvement per frame was "
                << (auxf_change / this->count_) << " over "
                << this->count_ << " frames.";
  if (auxf_final > auxf_orig) {
    *objf_impr = auxf_change;
    *count = this->count_;
    raw_fmllr_mat->CopyFromMat(fmllr_mat);
  } else {
    *objf_impr = 0.0;
    // don't update "raw_fmllr_mat"
  }
}

void FmllrRawAccs::SetZero() {
  count_ = 0.0;
  single_frame_stats_.count = 0.0;
  single_frame_stats_.s.SetZero();
  Q_.SetZero();
  S_.SetZero();
}

// Compute the M_i quantities, needed in the update.  This function could be
// greatly speeded up but I don't think it's the limiting factor.
void FmllrRawAccs::ComputeM(std::vector<Matrix<double> > *M) const {
  int32 full_dim = FullDim(), raw_dim = RawDim(),
      raw_dim2 = raw_dim * (raw_dim + 1);
  M->resize(full_dim);
  for (int32 i = 0; i < full_dim; i++)
    (*M)[i].Resize(raw_dim2, full_dim + 1);  

  // the N's are simpler matrices from which we'll interpolate the M's.
  // In this loop we imagine w are computing the vector of N's, but
  // when we get each element, if it's nonzero we propagate it straight
  // to the M's.
  for (int32 i = 0; i < full_dim; i++) {
    // i is index after fMLLR transform; i1 is splicing index,
    // i2 is cepstral index.
    int32 i1 = i / raw_dim, i2 = i % raw_dim;
    for (int32 j = 0; j < raw_dim2; j++) {
      // j1 is row-index of fMLLR transform, j2 is column-index
      int32 j1 = j / (raw_dim + 1), j2 = j % (raw_dim + 1);
      for (int32 k = 0; k < full_dim + 1; k++) {
        BaseFloat n_ijk;
        if (j1 != i2) {
          n_ijk = 0.0;
        } else if (k == full_dim) {
          if (j2 == raw_dim) // offset term in fMLLR matrix.
            n_ijk = 1.0;
          else
            n_ijk = 0.0;
        } else {
          // k1 is splicing index, k2 is cepstral idnex.
          int32 k1 = k / raw_dim, k2 = k % raw_dim;
          if (k1 != i1 || k2 != j2)
            n_ijk = 0.0;
          else
            n_ijk = 1.0;
        }
        if (n_ijk != 0.0)
          for (int32 l = 0; l < full_dim; l++)
            (*M)[l](j, k) += n_ijk * full_transform_(l, i);
      }
    }
  }
}

void FmllrRawAccs::ConvertToSimpleStats(
    Vector<double> *simple_linear_stats,
    SpMatrix<double> *simple_quadratic_stats) const {
  std::vector<Matrix<double> > M;
  ComputeM(&M);

  int32 full_dim = FullDim(), raw_dim = RawDim(), model_dim = ModelDim(),
      raw_dim2 = raw_dim * (raw_dim + 1),
      full_dim2 = ((full_dim+1)*(full_dim+2))/2;
  simple_linear_stats->Resize(raw_dim2);
  simple_quadratic_stats->Resize(raw_dim2);
  for (int32 i = 0; i < full_dim; i++) {
    Vector<double> q_i(full_dim + 1);
    SpMatrix<double> S_i(full_dim + 1);
    SubVector<double> S_i_vec(S_i.Data(), full_dim2);
    if (i < model_dim) {
      q_i.CopyFromVec(Q_.Row(i));
      S_i_vec.CopyFromVec(S_.Row(i));
    } else {
      q_i.CopyFromVec(Q_.Row(model_dim)); // The last row contains stats proportional
      // to "count", which we need to modify to be correct.
      q_i.Scale(-transform_offset_(i)); // These stats are zero (corresponding to
      // a zero-mean model) if there is no offset in the LDA transform.  Note:
      // the two statements above are the equivalent, for the rejected dims,
      // of the statement "a.AddVecVec(-1.0, b, offset);" for the kept ones.
      // 
      S_i_vec.CopyFromVec(S_.Row(model_dim)); // these are correct, and
      // all the same (corresponds to unit variance).
    }
    // The equation v = \sum_i M_i q_i:
    simple_linear_stats->AddMatVec(1.0, M[i], kNoTrans, q_i, 1.0);
    // The equation W = \sum_i M_i S_i M_i^T
    // Here, M[i] is quite sparse, so AddSmat2Sp will be faster.
    simple_quadratic_stats->AddSmat2Sp(1.0, M[i], kNoTrans, S_i, 1.0);
  }
}

// See header for comment.
void FmllrRawAccs::ConvertToPerRowStats(
    const Vector<double> &simple_linear_stats,
    const SpMatrix<double> &simple_quadratic_stats_sp,
    Matrix<double> *linear_stats,
    std::vector<SpMatrix<double> > *diag_stats,
    std::vector<std::vector<Matrix<double> > > *off_diag_stats) const {

  // get it as a Matrix, which makes it easier to extract sub-parts.
  Matrix<double> simple_quadratic_stats(simple_quadratic_stats_sp);

  linear_stats->Resize(RawDim(), RawDim() + 1);
  linear_stats->CopyRowsFromVec(simple_linear_stats);
  diag_stats->resize(RawDim());
  off_diag_stats->resize(RawDim());

  // Set *diag_stats
  int32 rd1 = RawDim() + 1;
  for (int32 i = 0; i < RawDim(); i++) {
    SubMatrix<double> this_diag(simple_quadratic_stats,
                                i * rd1, rd1,
                                i * rd1, rd1);
    (*diag_stats)[i].Resize(RawDim() + 1);
    (*diag_stats)[i].CopyFromMat(this_diag, kTakeMean);
  }    
  
  for (int32 i = 0; i < RawDim(); i++) {
    (*off_diag_stats)[i].resize(i);
    for (int32 j = 0; j < i; j++) {
      SubMatrix<double> this_off_diag(simple_quadratic_stats,
                                      i * rd1, rd1,
                                      j * rd1, rd1);
      (*off_diag_stats)[i][j] = this_off_diag;
    }
  }
}

double FmllrRawAccs::GetAuxf(const Vector<double> &simple_linear_stats,
                             const SpMatrix<double> &simple_quadratic_stats,
                             const Matrix<double> &fmllr_mat) const {
  // linearize transform...
  int32 raw_dim = RawDim(), spice_width = SpliceWidth();
  Vector<double> fmllr_vec(raw_dim * (raw_dim + 1));
  fmllr_vec.CopyRowsFromMat(fmllr_mat);
  SubMatrix<double> square_part(fmllr_mat, 0, raw_dim,
                                0, raw_dim);
  double logdet = square_part.LogDet();
  return VecVec(fmllr_vec, simple_linear_stats) -
      0.5 * VecSpVec(fmllr_vec, simple_quadratic_stats, fmllr_vec) +
      logdet * spice_width * count_;
}



} // namespace kaldi
