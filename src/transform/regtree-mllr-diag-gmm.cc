// transform/regtree-mllr-diag-gmm.cc

// Copyright 2009-2011  Saarland University;  Jan Silovsky

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
using std::pair;
#include <vector>
using std::vector;

#include "transform/regtree-mllr-diag-gmm.h"

namespace kaldi {

void RegtreeMllrDiagGmm::Init(int32 num_xforms, int32 dim) {
  if (num_xforms == 0) {  // empty transform
    xform_matrices_.clear();
    dim_ = 0;  // non-zero dimension is meaningless with empty transform
    num_xforms_ = 0;
    bclass2xforms_.clear();
  } else {
    KALDI_ASSERT(dim != 0);  // if not empty, dim = 0 is meaningless
    dim_ = dim;
    num_xforms_ = num_xforms;
    xform_matrices_.resize(num_xforms);
    vector< Matrix<BaseFloat> >::iterator xform_itr = xform_matrices_.begin(),
                                      xform_itr_end = xform_matrices_.end();
    for (; xform_itr != xform_itr_end; ++xform_itr) {
      xform_itr->Resize(dim, dim+1);
      xform_itr->SetUnit();
    }
  }
}

void RegtreeMllrDiagGmm::SetUnit() {
  vector< Matrix<BaseFloat> >::iterator xform_itr = xform_matrices_.begin(),
                                    xform_itr_end = xform_matrices_.end();
  for (; xform_itr != xform_itr_end; ++xform_itr) {
    xform_itr->SetUnit();
  }
}

void RegtreeMllrDiagGmm::TransformModel(const RegressionTree &regtree,
                                        AmDiagGmm *am) {
  KALDI_ASSERT(static_cast<int32>(bclass2xforms_.size()) ==
               regtree.NumBaseclasses());
  Vector<BaseFloat> extended_mean(dim_+1), xformed_mean(dim_);
  for (int32 bclass_index = 0, num_bclasses = regtree.NumBaseclasses();
       bclass_index < num_bclasses; ++bclass_index) {
    int32 xform_index;
    if ((xform_index = bclass2xforms_[bclass_index]) > -1) {
      KALDI_ASSERT(xform_index < num_xforms_);
      const vector< pair<int32, int32> > &bclass =
          regtree.GetBaseclass(bclass_index);
      for (vector< pair<int32, int32> >::const_iterator itr = bclass.begin(),
          end = bclass.end(); itr != end; ++itr) {
        SubVector<BaseFloat> tmp_mean(extended_mean.Range(0, dim_));
        am->GetGaussianMean(itr->first, itr->second, &tmp_mean);
        extended_mean(dim_) = 1.0;
        xformed_mean.AddMatVec(1.0, xform_matrices_[xform_index], kNoTrans,
                               extended_mean, 0.0);
        am->SetGaussianMean(itr->first, itr->second, xformed_mean);
      }  // end iterating over Gaussians in baseclass
    }  // else keep the means untransformed
  }  // end iterating over all baseclasses
  am->ComputeGconsts();
}


void RegtreeMllrDiagGmm::GetTransformedMeans(const RegressionTree &regtree,
                                             const AmDiagGmm &am,
                                             int32 pdf_index,
                                             MatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(static_cast<int32>(bclass2xforms_.size()) ==
               regtree.NumBaseclasses());
  int32 num_gauss = am.GetPdf(pdf_index).NumGauss();
  KALDI_ASSERT(out->NumRows() == num_gauss && out->NumCols() == dim_);

  Vector<BaseFloat> extended_mean(dim_+1);
  extended_mean(dim_) = 1.0;

  for (int32 gauss_index = 0; gauss_index < num_gauss; gauss_index++) {
    int32 bclass_index = regtree.Gauss2BaseclassId(pdf_index, gauss_index);
    int32 xform_index = bclass2xforms_[bclass_index];
    if (xform_index > -1) {  // use a transform
      KALDI_ASSERT(xform_index < num_xforms_);
      SubVector<BaseFloat> tmp_mean(extended_mean.Range(0, dim_));
      am.GetGaussianMean(pdf_index, gauss_index, &tmp_mean);
      SubVector<BaseFloat> out_row(out->Row(gauss_index));
      out_row.AddMatVec(1.0, xform_matrices_[xform_index], kNoTrans,
                        extended_mean, 0.0);
    } else {  // Copy untransformed mean
      SubVector<BaseFloat> out_row(out->Row(gauss_index));
      am.GetGaussianMean(pdf_index, gauss_index, &out_row);
    }
  }
}


void RegtreeMllrDiagGmm::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<MLLRXFORM>");
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
  WriteToken(out, binary, "</MLLRXFORM>");
}


void RegtreeMllrDiagGmm::Read(std::istream &in, bool binary) {
  ExpectToken(in, binary, "<MLLRXFORM>");
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
  ExpectToken(in, binary, "</MLLRXFORM>");
}

// ************************************************************************

void RegtreeMllrDiagGmmAccs::Init(int32 num_bclass, int32 dim) {
  if (num_bclass == 0) {  // empty stats
    DeletePointers(&baseclass_stats_);
    baseclass_stats_.clear();
    num_baseclasses_ = 0;
    dim_ = 0;  // non-zero dimension is meaningless in empty stats
  } else {
    KALDI_ASSERT(dim != 0);  // if not empty, dim = 0 is meaningless
    num_baseclasses_ = num_bclass;
    dim_ = dim;
    baseclass_stats_.resize(num_baseclasses_);
    for (vector<AffineXformStats*>::iterator it = baseclass_stats_.begin(),
        end = baseclass_stats_.end(); it != end; ++it) {
      *it = new AffineXformStats();
      (*it)->Init(dim_, dim_);
    }
  }
}

void RegtreeMllrDiagGmmAccs::SetZero() {
  for (vector<AffineXformStats*>::iterator it = baseclass_stats_.begin(),
      end = baseclass_stats_.end(); it != end; ++it) {
    (*it)->SetZero();
  }
}

BaseFloat RegtreeMllrDiagGmmAccs::AccumulateForGmm(
    const RegressionTree &regtree, const AmDiagGmm &am,
    const VectorBase<BaseFloat> &data, int32 pdf_index, BaseFloat weight) {
  const DiagGmm &pdf = am.GetPdf(pdf_index);
  int32 num_comp = static_cast<int32>(pdf.NumGauss());
  Vector<BaseFloat> posterior(num_comp);
  BaseFloat loglike = pdf.ComponentPosteriors(data, &posterior);
  posterior.Scale(weight);
  Vector<double> posterior_d(posterior);

  Vector<double> data_d(data);
  Vector<double> inv_var_x(dim_);
  Vector<double> extended_mean(dim_+1);
  SpMatrix<double> mean_scatter(dim_+1);

  for (int32 m = 0; m < num_comp; m++) {
    unsigned bclass = regtree.Gauss2BaseclassId(pdf_index, m);
    inv_var_x.CopyFromVec(pdf.inv_vars().Row(m));
    inv_var_x.MulElements(data_d);

    // Using SubVector to stop compiler warning
    SubVector<double> tmp_mean(extended_mean, 0, dim_);
    pdf.GetComponentMean(m, &tmp_mean);  // modifies extended_mean
    extended_mean(dim_) = 1.0;
    mean_scatter.SetZero();
    mean_scatter.AddVec2(1.0, extended_mean);

    baseclass_stats_[bclass]->beta_ += posterior_d(m);
    baseclass_stats_[bclass]->K_.AddVecVec(posterior_d(m), inv_var_x,
                                           extended_mean);
    vector< SpMatrix<double> > &G = baseclass_stats_[bclass]->G_;
    for (int32 d = 0; d < dim_; d++)
      G[d].AddSp((posterior_d(m) * pdf.inv_vars()(m, d)), mean_scatter);
  }
  return loglike;
}

void RegtreeMllrDiagGmmAccs::AccumulateForGaussian(
    const RegressionTree &regtree, const AmDiagGmm &am,
    const VectorBase<BaseFloat> &data, int32 pdf_index, int32 gauss_index,
    BaseFloat weight) {
  const DiagGmm &pdf = am.GetPdf(pdf_index);
  Vector<double> data_d(data);
  Vector<double> inv_var_x(dim_);
  Vector<double> extended_mean(dim_+1);
  double weight_d = static_cast<double>(weight);

  unsigned bclass = regtree.Gauss2BaseclassId(pdf_index, gauss_index);
  inv_var_x.CopyFromVec(pdf.inv_vars().Row(gauss_index));
  inv_var_x.MulElements(data_d);

  // Using SubVector to stop compiler warning
  SubVector<double> tmp_mean(extended_mean, 0, dim_);
  pdf.GetComponentMean(gauss_index, &tmp_mean);  // modifies extended_mean
  extended_mean(dim_) = 1.0;
  SpMatrix<double> mean_scatter(dim_+1);
  mean_scatter.AddVec2(1.0, extended_mean);

  baseclass_stats_[bclass]->beta_ += weight_d;
  baseclass_stats_[bclass]->K_.AddVecVec(weight_d, inv_var_x, extended_mean);
  vector< SpMatrix<double> > &G = baseclass_stats_[bclass]->G_;
  for (int32 d = 0; d < dim_; d++)
    G[d].AddSp((weight_d * pdf.inv_vars()(gauss_index, d)), mean_scatter);
}

void RegtreeMllrDiagGmmAccs::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<MLLRACCS>");
  WriteToken(out, binary, "<NUMBASECLASSES>");
  WriteBasicType(out, binary, num_baseclasses_);
  WriteToken(out, binary, "<DIMENSION>");
  WriteBasicType(out, binary, dim_);
  WriteToken(out, binary, "<STATS>");
  vector<AffineXformStats*>::const_iterator itr = baseclass_stats_.begin(),
                                            end = baseclass_stats_.end();
  for ( ; itr != end; ++itr)
    (*itr)->Write(out, binary);
  WriteToken(out, binary, "</MLLRACCS>");
}

void RegtreeMllrDiagGmmAccs::Read(std::istream &in, bool binary, bool add) {
  ExpectToken(in, binary, "<MLLRACCS>");
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
  ExpectToken(in, binary, "</MLLRACCS>");
}

static void ComputeMllrMatrix(const Matrix<double> &K,
                              const vector< SpMatrix<double> > &G,
                              Matrix<BaseFloat> *out) {
  int32 dim = G.size();
  Matrix<double> tmp_out(dim, dim+1);
  for (int32 d = 0; d < dim; d++) {
    if (G[d].Cond() > 1.0e+9) {
      KALDI_WARN << "Dim " << d << ": Badly conditioned stats. Setting MLLR "
                 << "transform to unit.";
      tmp_out.SetUnit();
      break;
    }
    SpMatrix<double> inv_g(G[d]);
//    KALDI_LOG << "Dim " << d << ": G: max = " << inv_g.Max() << ", min = "
//              << inv_g.Min() << ", log det = " << inv_g.LogDet(NULL)
//              << ", cond = " << inv_g.Cond();
    inv_g.Invert();
//    KALDI_LOG << "Inv G: max = " << inv_g.Max() << ", min = " << inv_g.Min()
//              << ", log det = " << inv_g.LogDet(NULL) << ", cond = "
//              << inv_g.Cond();
    tmp_out.Row(d).AddSpVec(1.0, inv_g, K.Row(d), 0.0);
  }
  out->CopyFromMat(tmp_out, kNoTrans);
}

static BaseFloat MllrAuxFunction(const Matrix<BaseFloat> &xform,
                                 const AffineXformStats &stats) {
  int32 dim = stats.G_.size();
  Matrix<double> xform_d(xform);
  Vector<double> xform_row_g(dim + 1);
  SubMatrix<double> A(xform_d, 0, dim, 0, dim);
  double obj = TraceMatMat(xform_d, stats.K_, kTrans);
  for (int32 d = 0; d < dim; d++) {
    xform_row_g.AddSpVec(1.0, stats.G_[d], xform_d.Row(d), 0.0);
    obj -= 0.5 * VecVec(xform_row_g, xform_d.Row(d));
  }
  return obj;
}

void RegtreeMllrDiagGmmAccs::Update(const RegressionTree &regtree,
                                    const RegtreeMllrOptions &opts,
                                    RegtreeMllrDiagGmm *out_mllr,
                                    BaseFloat *auxf_impr,
                                    BaseFloat *t) const {
  BaseFloat tot_auxf_impr = 0, tot_t = 0;
  Matrix<BaseFloat> xform_mat(dim_, dim_ + 1);
  if (opts.use_regtree) {  // estimate transforms using a regression tree
    vector<AffineXformStats*> regclass_stats;
    vector<int32> base2regclass;
    bool update_xforms = regtree.GatherStats(baseclass_stats_, opts.min_count,
                                             &base2regclass, &regclass_stats);
    out_mllr->set_bclass2xforms(base2regclass);
    // If update_xforms == true, none should be negative, else all should be -1
    if (update_xforms) {
      out_mllr->Init(regclass_stats.size(), dim_);
      for (int32 rclass_index = 0, num_rclass = regclass_stats.size();
           rclass_index < num_rclass; ++rclass_index) {
        KALDI_ASSERT(regclass_stats[rclass_index]->beta_ >= opts.min_count);
        xform_mat.SetUnit();
        BaseFloat obj_old = MllrAuxFunction(xform_mat,
                                            *(regclass_stats[rclass_index]));
        ComputeMllrMatrix(regclass_stats[rclass_index]->K_,
                          regclass_stats[rclass_index]->G_, &xform_mat);
        out_mllr->SetParameters(xform_mat, rclass_index);
        BaseFloat obj_new = MllrAuxFunction(xform_mat,
                                            *(regclass_stats[rclass_index]));
        KALDI_LOG << "MLLR: regclass " << (rclass_index)
                  << ": Objective function impr per frame is "
                  << ((obj_new - obj_old)/regclass_stats[rclass_index]->beta_)
                  << " over " << regclass_stats[rclass_index]->beta_
                  << " frames.";
        KALDI_ASSERT(obj_new >= obj_old - (std::abs(obj_new)+std::abs(obj_old))*1.0e-05);
        tot_t += regclass_stats[rclass_index]->beta_;
        tot_auxf_impr += obj_new - obj_old;
      }
    } else {
      out_mllr->Init(1, dim_);  // Use a unit transform at the root.
    }
    DeletePointers(&regclass_stats);
    // end of estimation using regression tree
  } else {  // estimate 1 transform per baseclass (if enough count)
    out_mllr->Init(num_baseclasses_, dim_);
    vector<int32> base2xforms(num_baseclasses_, -1);
    for (int32 bclass_index = 0; bclass_index < num_baseclasses_;
         ++bclass_index) {
      if (baseclass_stats_[bclass_index]->beta_ > opts.min_count) {
        base2xforms[bclass_index] = bclass_index;
        xform_mat.SetUnit();
        BaseFloat obj_old = MllrAuxFunction(xform_mat,
                                            *(baseclass_stats_[bclass_index]));
        ComputeMllrMatrix(baseclass_stats_[bclass_index]->K_,
                          baseclass_stats_[bclass_index]->G_, &xform_mat);
        out_mllr->SetParameters(xform_mat, bclass_index);
        BaseFloat obj_new = MllrAuxFunction(xform_mat,
                                            *(baseclass_stats_[bclass_index]));
        KALDI_LOG << "MLLR: base-class " << (bclass_index)
                  << ": Auxiliary function impr per frame is "
                  << ((obj_new-obj_old)/baseclass_stats_[bclass_index]->beta_);
        KALDI_ASSERT(obj_new >= obj_old - (std::abs(obj_new)+std::abs(obj_old))*1.0e-05);
        tot_t += baseclass_stats_[bclass_index]->beta_;
        tot_auxf_impr += obj_new - obj_old;
      } else {
        KALDI_WARN << "For baseclass "  << (bclass_index) << " count = "
                   << (baseclass_stats_[bclass_index]->beta_) << " < "
                   << opts.min_count << ": not updating MLLR";
        tot_t += baseclass_stats_[bclass_index]->beta_;
      }
    }  // end looping over all baseclasses
    out_mllr->set_bclass2xforms(base2xforms);
  }  // end of estimating one transform per baseclass
  if (auxf_impr != NULL) *auxf_impr = tot_auxf_impr;
  if (t != NULL) *t = tot_t;
}

}  // namespace kaldi

