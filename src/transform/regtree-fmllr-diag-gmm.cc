// transform/regtree-fmllr-diag-gmm.cc

// Copyright 2009-2011  Saarland University;  Georg Stemmer;
//                      Microsoft Corporation

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

#include "itf/optimizable-itf.h"
#include "transform/fmllr-diag-gmm.h"
#include "transform/regtree-fmllr-diag-gmm.h"

namespace kaldi {

void RegtreeFmllrDiagGmm::Init(size_t num_xforms, size_t dim) {
  if (num_xforms == 0) {  // empty transform
    xform_matrices_.clear();
    logdet_.Resize(0);
    valid_logdet_ = false;
    dim_ = 0;  // non-zero dimension is meaningless with empty transform
    num_xforms_ = 0;
  } else {
    KALDI_ASSERT(dim != 0);  // if not empty, dim = 0 is meaningless
    dim_ = dim;
    num_xforms_ = num_xforms;
    xform_matrices_.resize(num_xforms);
    logdet_.Resize(num_xforms);
    vector< Matrix<BaseFloat> >::iterator xform_itr = xform_matrices_.begin(),
        xform_itr_end = xform_matrices_.end();
    for (; xform_itr != xform_itr_end; ++xform_itr) {
      xform_itr->Resize(dim, dim+1);
      xform_itr->SetUnit();
    }
    valid_logdet_ = true;
  }
}

void RegtreeFmllrDiagGmm::SetUnit() {
  KALDI_ASSERT(num_xforms_ > 0 && dim_ > 0);
  vector< Matrix<BaseFloat> >::iterator xform_itr = xform_matrices_.begin(),
      xform_itr_end = xform_matrices_.end();
  for (; xform_itr != xform_itr_end; ++xform_itr) {
    xform_itr->SetUnit();
  }
}

void RegtreeFmllrDiagGmm::Validate() {
  if (dim_ < 0 || num_xforms_ < 0) {  // uninitialized case
    KALDI_ERR <<"Do not call Validate() with an uninitialized object (dim = "
              << (dim_) << ", # transforms = " << (num_xforms_);
  } else if (dim_ * num_xforms_ == 0) {  // empty case
    KALDI_ASSERT(num_xforms_ == 0 && dim_ == 0);
    if (xform_matrices_.size() != 0 || logdet_.Dim() != 0) {
      KALDI_ERR << "Number of transforms = " << (xform_matrices_.size())
                << ", number of log-determinant terms = " << (logdet_.Dim())
                << ". Expected number = 0";
    }
    return;
  }

  // non-empty case: typical usage scenario
  if (xform_matrices_.size() != static_cast<size_t>(num_xforms_)
      || logdet_.Dim() != num_xforms_) {
    KALDI_ERR << "Number of transforms = " << (xform_matrices_.size())
              << ", number of log-determinant terms = " << (logdet_.Dim())
              << ". `Expected number = " << (num_xforms_);
  }
  for (int32 i = 0; i < num_xforms_; i++) {
    if (xform_matrices_[i].NumRows() != dim_ ||
        xform_matrices_[i].NumCols() != (dim_+1)) {
      KALDI_ERR << "For transform " << (i) << ": inconsistent size: rows = "
                << (xform_matrices_[i].NumRows()) << ", cols = "
                << xform_matrices_[i].NumCols() << ", dim = " << (dim_);
    }
  }
  if (bclass2xforms_.size() > 0) {
    for (int32 i = 0, maxi = bclass2xforms_.size(); i < maxi; i++) {
      if (bclass2xforms_[i] >= num_xforms_) {
        KALDI_ERR << "For baseclass " << (i) << ", transform index "
                  << (bclass2xforms_[i]) << " exceeds total transforms "
                  << (num_xforms_);
      }
    }
  } else {
    if (num_xforms_ > 1) {
      KALDI_WARN << "Multiple FMLLR transforms found without baseclass info.";
    }
  }
}

void RegtreeFmllrDiagGmm::ComputeLogDets() {
  logdet_.Resize(num_xforms_);
  for (int32 r = 0; r < num_xforms_; r++) {
    SubMatrix<BaseFloat> tmp_a(xform_matrices_[r], 0, dim_, 0,
                               dim_);
    logdet_(r) = tmp_a.LogDet();
    KALDI_ASSERT(!KALDI_ISNAN(logdet_(r)));
  }
  valid_logdet_ = true;
}

void RegtreeFmllrDiagGmm::TransformFeature(const VectorBase<BaseFloat> &in,
                                    vector<Vector<BaseFloat> > *out) const {
  KALDI_ASSERT(out != NULL);

  if (xform_matrices_.size() == 0) {  // empty transform
    KALDI_ASSERT(num_xforms_ == 0 && dim_ == 0 && logdet_.Dim() == 0);
    KALDI_WARN << "Asked to apply empty feature transform. Copying instead.";
    out->resize(1);
    (*out)[0].Resize(in.Dim());
    (*out)[0].CopyFromVec(in);
    return;
  } else {
    KALDI_ASSERT(in.Dim() == dim_);
    // if (!valid_logdet_)
    // KALDI_ERR << "Must call ComputeLogDets() before transforming data.";
    // [no need for this check].
    Vector<BaseFloat> extended_feat(dim_ + 1);
    extended_feat.Range(0, dim_).CopyFromVec(in);
    extended_feat(dim_) = 1.0;
    KALDI_ASSERT(num_xforms_ > 0);
    out->resize(num_xforms_);
    for (int32 xform_index = 0; xform_index < num_xforms_;
         ++xform_index) {
      (*out)[xform_index].Resize(dim_);
      (*out)[xform_index].AddMatVec(1.0, xform_matrices_[xform_index],
                                    kNoTrans, extended_feat, 0.0);
    }
  }
}

void RegtreeFmllrDiagGmm::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<FMLLRXFORM>");
  WriteToken(out, binary, "<NUMXFORMS>");
  WriteBasicType(out, binary, num_xforms_);
  WriteToken(out, binary, "<DIMENSION>");
  WriteBasicType(out, binary, dim_);

  vector< Matrix<BaseFloat> >::const_iterator xform_itr =
      xform_matrices_.begin(), xform_itr_end = xform_matrices_.end();
  for (; xform_itr != xform_itr_end; ++xform_itr) {
    WriteToken(out, binary, "<XFORM>");
    xform_itr->Write(out, binary);
  }

  WriteToken(out, binary, "<BCLASS2XFORMS>");
  WriteIntegerVector(out, binary, bclass2xforms_);
  WriteToken(out, binary, "</FMLLRXFORM>");
}


void RegtreeFmllrDiagGmm::Read(std::istream &in, bool binary) {
  ExpectToken(in, binary, "<FMLLRXFORM>");
  ExpectToken(in, binary, "<NUMXFORMS>");
  ReadBasicType(in, binary, &num_xforms_);
  ExpectToken(in, binary, "<DIMENSION>");
  ReadBasicType(in, binary, &dim_);
  KALDI_ASSERT(num_xforms_ >= 0 && dim_ >= 0);  // can be 0 for empty xform

  xform_matrices_.resize(num_xforms_);
  vector< Matrix<BaseFloat> >::iterator xform_itr = xform_matrices_.begin(),
      xform_itr_end = xform_matrices_.end();
  for (; xform_itr != xform_itr_end; ++xform_itr) {
    ExpectToken(in, binary, "<XFORM>");
    xform_itr->Read(in, binary);
    KALDI_ASSERT(xform_itr->NumRows() == (xform_itr->NumCols() - 1)
           && xform_itr->NumRows() == dim_);
  }

  ExpectToken(in, binary, "<BCLASS2XFORMS>");
  ReadIntegerVector(in, binary, &bclass2xforms_);
  ExpectToken(in, binary, "</FMLLRXFORM>");
  ComputeLogDets();  // so that the transforms can be used.
}

// ************************************************************************




void RegtreeFmllrDiagGmmAccs::Init(size_t num_bclass, size_t dim) {
  if (num_bclass == 0) {  // empty stats
    DeletePointers(&baseclass_stats_);
    baseclass_stats_.clear();
    num_baseclasses_ = 0;
    dim_ = 0;  // non-zero dimension is meaningless in empty stats
  } else {
    KALDI_ASSERT(dim != 0);  // if not empty, dim = 0 is meaningless
    num_baseclasses_ = num_bclass;
    dim_ = dim;
    DeletePointers(&baseclass_stats_);
    baseclass_stats_.resize(num_bclass);
    for (vector<AffineXformStats*>::iterator it = baseclass_stats_.begin(),
             end = baseclass_stats_.end(); it != end; ++it) {
      *it = new AffineXformStats();
      (*it)->Init(dim, dim);
    }
  }
}

void RegtreeFmllrDiagGmmAccs::SetZero() {
  for (vector<AffineXformStats*>::iterator it = baseclass_stats_.begin(),
           end = baseclass_stats_.end(); it != end; ++it) {
    (*it)->SetZero();
  }
}

BaseFloat RegtreeFmllrDiagGmmAccs::AccumulateForGmm(
    const RegressionTree &regtree, const AmDiagGmm &am,
    const VectorBase<BaseFloat> &data, size_t pdf_index, BaseFloat weight) {
  const DiagGmm &pdf = am.GetPdf(pdf_index);
  int32 num_comp = pdf.NumGauss();
  Vector<BaseFloat> posterior(num_comp);
  BaseFloat loglike = pdf.ComponentPosteriors(data, &posterior);
  posterior.Scale(weight);
  Vector<double> posterior_d(posterior);

  Vector<double> extended_data(dim_+1);
  extended_data.Range(0, dim_).CopyFromVec(data);
  extended_data(dim_) = 1.0;
  SpMatrix<double> scatter(dim_+1);
  scatter.AddVec2(1.0, extended_data);

  Vector<double> inv_var_mean(dim_);
  Matrix<double> g_scale(baseclass_stats_.size(), dim_);  // scale on "scatter" for each dim.
  for (int32 m = 0; m < num_comp; m++) {
    inv_var_mean.CopyRowFromMat(pdf.means_invvars(), m);
    int32 bclass = regtree.Gauss2BaseclassId(pdf_index, m);

    baseclass_stats_[bclass]->beta_ += posterior_d(m);
    baseclass_stats_[bclass]->K_.AddVecVec(posterior_d(m), inv_var_mean,
                                           extended_data);
    for (int32 d = 0; d < dim_; d++)
      g_scale(bclass, d) +=  posterior(m) * pdf.inv_vars()(m, d);
  }
  for (size_t bclass = 0; bclass < baseclass_stats_.size(); bclass++) {
    vector< SpMatrix<double> > &G = baseclass_stats_[bclass]->G_;
    for (int32 d = 0; d < dim_; d++)
      if (g_scale(bclass, d) != 0.0)
        G[d].AddSp(g_scale(bclass, d), scatter);
  }
  return loglike;
}

void RegtreeFmllrDiagGmmAccs::AccumulateForGaussian(
    const RegressionTree &regtree, const AmDiagGmm &am,
    const VectorBase<BaseFloat> &data, size_t pdf_index, size_t gauss_index,
    BaseFloat weight) {
  const DiagGmm &pdf = am.GetPdf(pdf_index);
  size_t dim = static_cast<size_t>(dim_);
  Vector<double> extended_data(dim+1);
  extended_data.Range(0, dim).CopyFromVec(data);
  extended_data(dim) = 1.0;
  SpMatrix<double> scatter(dim+1);
  scatter.AddVec2(1.0, extended_data);
  double weight_d = static_cast<double>(weight);

  unsigned bclass = regtree.Gauss2BaseclassId(pdf_index, gauss_index);
  Vector<double> inv_var_mean(dim_);
  inv_var_mean.CopyRowFromMat(pdf.means_invvars(), gauss_index);

  baseclass_stats_[bclass]->beta_ += weight_d;
  baseclass_stats_[bclass]->K_.AddVecVec(weight_d, inv_var_mean, extended_data);
  vector< SpMatrix<double> > &G = baseclass_stats_[bclass]->G_;
  for (size_t d = 0; d < dim; d++)
    G[d].AddSp((weight_d * pdf.inv_vars()(gauss_index, d)), scatter);
}

void RegtreeFmllrDiagGmmAccs::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<FMLLRACCS>");
  WriteToken(out, binary, "<NUMBASECLASSES>");
  WriteBasicType(out, binary, num_baseclasses_);
  WriteToken(out, binary, "<DIMENSION>");
  WriteBasicType(out, binary, dim_);
  WriteToken(out, binary, "<STATS>");
  vector<AffineXformStats*>::const_iterator itr = baseclass_stats_.begin(),
      end = baseclass_stats_.end();
  for ( ; itr != end; ++itr)
    (*itr)->Write(out, binary);
  WriteToken(out, binary, "</FMLLRACCS>");
}

void RegtreeFmllrDiagGmmAccs::Read(std::istream &in, bool binary, bool add) {
  ExpectToken(in, binary, "<FMLLRACCS>");
  ExpectToken(in, binary, "<NUMBASECLASSES>");
  ReadBasicType(in, binary, &num_baseclasses_);
  ExpectToken(in, binary, "<DIMENSION>");
  ReadBasicType(in, binary, &dim_);
  KALDI_ASSERT(num_baseclasses_ > 0 && dim_ > 0);
  baseclass_stats_.resize(num_baseclasses_);
  ExpectToken(in, binary, "<STATS>");
  vector<AffineXformStats*>::iterator itr = baseclass_stats_.begin(),
      end = baseclass_stats_.end();
  for ( ; itr != end; ++itr) {
    *itr = new AffineXformStats();
    (*itr)->Init(dim_, dim_);
    (*itr)->Read(in, binary, add);
  }
  ExpectToken(in, binary, "</FMLLRACCS>");
}


void RegtreeFmllrDiagGmmAccs::Update(const RegressionTree &regtree,
                              const RegtreeFmllrOptions &opts,
                              RegtreeFmllrDiagGmm *out_fmllr,
                              BaseFloat *auxf_impr_out,
                              BaseFloat *tot_t_out) const {
  BaseFloat tot_auxf_impr = 0.0, tot_t = 0.0;
  Matrix<BaseFloat> xform_mat(dim_, dim_+1);
  if (opts.use_regtree) {  // estimate transforms using a regression tree
    vector<AffineXformStats*> regclass_stats;
    vector<int32> base2regclass;
    bool update_xforms = regtree.GatherStats(baseclass_stats_, opts.min_count,
                                             &base2regclass, &regclass_stats);
    out_fmllr->set_bclass2xforms(base2regclass);
    // If update_xforms == true, none should be negative, else all should be -1
    if (update_xforms) {
      out_fmllr->Init(regclass_stats.size(), dim_);
      size_t num_rclass = regclass_stats.size();
      for (size_t rclass_index = 0;
           rclass_index < num_rclass; ++rclass_index) {
        KALDI_ASSERT(regclass_stats[rclass_index]->beta_ >= opts.min_count);
        xform_mat.SetUnit();
        tot_t += regclass_stats[rclass_index]->beta_;

        tot_auxf_impr +=
            ComputeFmllrMatrixDiagGmmFull(xform_mat, *(regclass_stats[rclass_index]),
                                          opts.num_iters, &xform_mat);
        
        out_fmllr->SetParameters(xform_mat, rclass_index);
      }
      KALDI_LOG << "Estimated " << num_rclass << " regression classes.";
    } else {
      out_fmllr->Init(1, dim_);  // Use a unit transform at the root.
    }
    DeletePointers(&regclass_stats);
    // end of estimation using regression tree
  } else {  // No regtree: estimate 1 transform per baseclass (if enough count)
    for (int32 bclass_index = 0; bclass_index < num_baseclasses_;
         ++bclass_index) {
      tot_t += baseclass_stats_[bclass_index]->beta_;
    }

    out_fmllr->Init(num_baseclasses_, dim_);
    vector<int32> base2regclass(num_baseclasses_);
    for (int32 bclass_index = 0; bclass_index < num_baseclasses_;
         ++bclass_index) {
      if (baseclass_stats_[bclass_index]->beta_ >= opts.min_count) {
        xform_mat.SetUnit();

        if (opts.update_type == "full") {
          tot_auxf_impr +=
              ComputeFmllrMatrixDiagGmmFull(xform_mat,
                                            *(baseclass_stats_[bclass_index]),
                                            opts.num_iters, &xform_mat);
        } else if (opts.update_type == "diag")
          tot_auxf_impr +=
              ComputeFmllrMatrixDiagGmmDiagonal(xform_mat,
                                                *(baseclass_stats_[bclass_index]),
                                                &xform_mat);
        else if (opts.update_type == "offset")
          tot_auxf_impr +=
              ComputeFmllrMatrixDiagGmmOffset(xform_mat,
                                              *(baseclass_stats_[bclass_index]),
                                              &xform_mat);
        else if (opts.update_type == "none")
          tot_auxf_impr = 0.0;
        else
          KALDI_ERR << "Unknown fMLLR update type " << opts.update_type
                    << ", fmllr-update-type must be one of \"full\"|\"diag\"|\"offset\"|\"none\"";

        out_fmllr->SetParameters(xform_mat, bclass_index);
        base2regclass[bclass_index] = bclass_index;
      } else {
        KALDI_WARN << "For baseclass " << (bclass_index) << " count = "
                   << (baseclass_stats_[bclass_index]->beta_) << " < "
                   << opts.min_count << ": not updating FMLLR";
        base2regclass[bclass_index] = -1;
      }
      out_fmllr->set_bclass2xforms(base2regclass);
    }  // end looping over all baseclasses
  }  // end of estimating one transform per baseclass without regtree
  if (auxf_impr_out) *auxf_impr_out = tot_auxf_impr;
  if (tot_t_out) *tot_t_out = tot_t;
}




}  // namespace kaldi

