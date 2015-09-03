// transform/hlda.cc

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.;  Georg Stemmer

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
#include "util/common-utils.h"
#include "transform/hlda.h"
#include "transform/mllt.h"

namespace kaldi {


void HldaAccsDiagGmm::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<HldaAccsDiagGmm>");
  ExpectToken(is, binary, "<S>");
  int32 dim;  // just the #elems of S_, equals model-dim+1.
  ReadBasicType(is, binary, &dim);
  if (add && S_.size() != 0 && static_cast<size_t>(dim) != S_.size())
    KALDI_ERR << "HldaAccsDiagGmm::Read, summing accs of different size.";
  if (!add || S_.empty()) S_.resize(dim);
  for (size_t i = 0; i < S_.size(); i++)
    S_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<occs>");
  int32 npdfs;
  ReadBasicType(is, binary, &npdfs);
  if (add && occs_.size() != 0 && static_cast<size_t>(npdfs) != occs_.size())
    KALDI_ERR << "HldaAccsDiagGmm::Read, summing accs of different size.";
  if (!add || occs_.empty()) {
    occs_.resize(npdfs);
    mean_accs_.resize(npdfs);
  }
  for (size_t i = 0; i < occs_.size(); i++)
    occs_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<mean_accs>");
  for (size_t i = 0; i < mean_accs_.size(); i++)
    mean_accs_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<speedup>");
  ReadBasicType(is, binary, &speedup_);
  if (speedup_ != 1.0) {
    if (!add || occs_sub_.empty()) {
      occs_sub_.resize(npdfs);
      mean_accs_sub_.resize(npdfs);
    }
    ExpectToken(is, binary, "<occs_sub>");
    for (size_t i = 0; i < occs_sub_.size(); i++)
      occs_sub_[i].Read(is, binary, add);
    ExpectToken(is, binary, "<mean_accs_sub>");
    for (size_t i = 0; i < mean_accs_sub_.size(); i++)
      mean_accs_sub_[i].Read(is, binary, add);
  }

  ExpectToken(is, binary, "<sample_gconst>");
  ReadBasicType(is, binary, &sample_gconst_);
  ExpectToken(is, binary, "</HldaAccsDiagGmm>");
}

void HldaAccsDiagGmm::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<HldaAccsDiagGmm>");
  WriteToken(os, binary, "<S>");
  int32 dim = S_.size();  // just the #elems of S_, equals model-dim+1.
  WriteBasicType(os, binary, dim);
  for (int32 i = 0; i < dim; i++) S_[i].Write(os, binary);
  KALDI_ASSERT(mean_accs_.size() == occs_.size());
  WriteToken(os, binary, "<occs>");
  int32 npdfs = occs_.size();
  WriteBasicType(os, binary, npdfs);
  for (int32 i = 0; i < npdfs; i++)
    occs_[i].Write(os, binary);
  WriteToken(os, binary, "<mean_accs>");
  for (int32 i = 0; i < npdfs; i++)
    mean_accs_[i].Write(os, binary);
  WriteToken(os, binary, "<speedup>");
  WriteBasicType(os, binary, speedup_);
  if (speedup_ != 1.0) {
    WriteToken(os, binary, "<occs_sub>");
    for (int32 i = 0; i < npdfs; i++)
      occs_sub_[i].Write(os, binary);
    WriteToken(os, binary, "<mean_accs_sub>");
    for (int32 i = 0; i < npdfs; i++)
      mean_accs_sub_[i].Write(os, binary);
  }
  WriteToken(os, binary, "<sample_gconst>");
  WriteBasicType(os, binary, sample_gconst_);
  WriteToken(os, binary, "</HldaAccsDiagGmm>");
}

void HldaAccsDiagGmm::Init(const AmDiagGmm &am,
                           int32 orig_feat_dim,
                           BaseFloat speedup) {
  KALDI_ASSERT(am.Dim() != 0);
  int32 num_pdfs = am.NumPdfs(), model_dim = am.Dim();
  KALDI_ASSERT(orig_feat_dim > 0 && orig_feat_dim >= model_dim);

  S_.resize(model_dim+1);
  for (int32 i = 0; i <= model_dim; i++)
    S_[i].Resize(orig_feat_dim);
  occs_.resize(num_pdfs);
  mean_accs_.resize(num_pdfs);
  for (int32 i = 0; i < num_pdfs; i++) {
    occs_[i].Resize(am.NumGaussInPdf(i));
    mean_accs_[i].Resize(am.NumGaussInPdf(i), orig_feat_dim);
  }
  speedup_ = speedup;
  if (speedup_ == 1.0) {
    occs_sub_.resize(0);
    mean_accs_sub_.resize(0);
  } else {
    occs_sub_.resize(num_pdfs);
    mean_accs_sub_.resize(num_pdfs);
    for (int32 i = 0; i < num_pdfs; i++) {
      occs_sub_[i].Resize(am.NumGaussInPdf(i));
      mean_accs_sub_[i].Resize(am.NumGaussInPdf(i), orig_feat_dim);
    }
  }

  sample_gconst_ = am.GetPdf(0).gconsts()(0);

}


void
HldaAccsDiagGmm::
AccumulateFromPosteriors(int32 pdf_id,
                         const DiagGmm &gmm,
                         const VectorBase<BaseFloat> &data,
                         const VectorBase<BaseFloat> &posteriors) {
  Vector<double> data_dbl(data);
  KALDI_ASSERT(static_cast<size_t>(pdf_id) < occs_.size()
               && occs_[pdf_id].Dim() == posteriors.Dim());
  KALDI_ASSERT(mean_accs_[pdf_id].NumCols() == data.Dim()
               && "Feature dim mismatch in HLDA computation ");
  double tot_occ = 0.0;
  int32 model_dim = S_.size() - 1;
  Vector<BaseFloat> tot_occ_times_inv_var(model_dim);

  if (speedup_ == 1.0) {  // no speedup; only one type of acc.
    for (int32 i = 0; i < posteriors.Dim(); i++) {
      if (posteriors(i) > 1.0e-05) {
        BaseFloat occ = posteriors(i);
        tot_occ += occ;
        occs_[pdf_id](i) += occ;
        mean_accs_[pdf_id].Row(i).AddVec(occ, data_dbl);

        SubVector<BaseFloat> inv_var(gmm.inv_vars(), i);  // this inv-var.
        tot_occ_times_inv_var.AddVec(occ, inv_var);
      }
    }
  } else {
    // Using a data subset.
    // In any case, accumulate regular occs and means.
    Vector<double> posteriors_dbl(posteriors);
    occs_[pdf_id].AddVec(1.0, posteriors_dbl);
    mean_accs_[pdf_id].AddVecVec(1.0, posteriors_dbl, data_dbl);
    if (RandUniform() > speedup_) return;  // continue with probability "speedup".

    for (int32 i = 0; i < posteriors.Dim(); i++) {
      if (posteriors(i) > 1.0e-05) {
        BaseFloat occ = posteriors(i);
        tot_occ += occ;
        occs_sub_[pdf_id](i) += occ;
        mean_accs_sub_[pdf_id].Row(i).AddVec(occ, data_dbl);

        SubVector<BaseFloat> inv_var(gmm.inv_vars(), i);  // this inv-var.
        tot_occ_times_inv_var.AddVec(occ, inv_var);
      }
    }

  }
  if (tot_occ != 0.0) {
    for (int32 i = 0; i < model_dim; i++)
      S_[i].AddVec2(tot_occ_times_inv_var(i), data_dbl);
    S_[model_dim].AddVec2(tot_occ, data_dbl);
  }
}


void HldaAccsDiagGmm::Update(AmDiagGmm *am,
                             MatrixBase<BaseFloat> *Mfull,
                             MatrixBase<BaseFloat> *M_out,
                             BaseFloat *objf_impr_out,
                             BaseFloat *count_out) const {
  KALDI_ASSERT(am != NULL && Mfull != NULL);
  KALDI_ASSERT(!S_.empty());

  if (!ApproxEqual(sample_gconst_, am->GetPdf(0).gconsts()(0), 1.0e-05)) {
    KALDI_ERR << "You have to call the HLDA update with the same model as used "
        "for accumulation.";
  }

  int32 model_dim = S_.size() - 1;
  KALDI_ASSERT(model_dim == am->Dim());

  int32 feat_dim = S_[0].NumRows();
  KALDI_ASSERT(feat_dim >= model_dim);

  KALDI_ASSERT(Mfull->NumRows() == feat_dim && Mfull->NumCols() == feat_dim);
  // this local G will be like the MLLT stats in a dimension equal
  // to feat_dim.
  std::vector<SpMatrix<double> > G(feat_dim);

  // This loop sets G to the outer product of the data, scaled
  // by inverse var.  Later we subtract the mean outer-product.
  for (int32 i = 0; i < feat_dim; i++) {
    G[i].Resize(feat_dim);
    if (i < model_dim) {
      G[i].CopyFromSp(S_[i]);
    } else {
      G[i].CopyFromSp(S_[model_dim]);  // unit variance in all the
      // rest of the dims, so we use the same stats.
    }
  }

  const std::vector<Vector<double> > &occs = (speedup_ == 1.0 ? occs_ : occs_sub_);
  const std::vector<Matrix<double> > &mean_accs = (speedup_ == 1.0 ? mean_accs_ :
                                                   mean_accs_sub_);

  int32 num_pdfs = occs.size();
  Vector<double> tot_mean_acc(feat_dim);
  double tot_occ = 0.0;  // will be occ of subset of data, if speedup_ != 1.0
  for (int32 p = 0; p < num_pdfs; p++) {
    int32 num_gauss = occs[p].Dim();
    const DiagGmm &gmm = am->GetPdf(p);
    KALDI_ASSERT(num_gauss == gmm.NumGauss());
    for (int32 g = 0; g < num_gauss; g++) {
      double occ = occs[p](g), inv_occ = (occ == 0.0 ? 0.0 : 1.0/occ);
      Vector<double> mean(feat_dim);
      mean.AddVec(inv_occ, mean_accs[p].Row(g));
      tot_mean_acc.AddVec(1.0, mean_accs[p].Row(g));
      tot_occ += occ;
      // update G matrices (subtracting outer-product of means, scaled by
      // occ and inverse-var); has same effect as if G is summed outer product of
      // (x-mu)^2, scaled by occ and inverse-var.

      SubVector<BaseFloat> inv_var(gmm.inv_vars(), g);  // this inv-var.
      for (int32 d = 0; d < model_dim; d++) {
        G[d].AddVec2(-1.0*occ*inv_var(d), mean);
      }
    }
  }
  KALDI_ASSERT(tot_occ > 0.0);
  Vector<double> tot_mean(tot_mean_acc);
  tot_mean.Scale(1.0 / tot_occ);
  // subtract total occ times outer product of global mean, from
  // dimensions of G that correspond to "rejected dimensions"
  // (with unit-var, global mean).
  for (int32 d = model_dim; d < feat_dim; d++)
    G[d].AddVec2(-tot_occ, tot_mean);

  for (int32 d = 0; d < feat_dim; d++)
    KALDI_ASSERT(G[d].IsPosDef());

  MlltAccs::Update(tot_occ, G, Mfull, objf_impr_out, count_out);

  SubMatrix<BaseFloat> Mpart(*Mfull, 0, model_dim, 0, feat_dim);
  if (M_out) {
    KALDI_ASSERT(M_out->NumRows() == model_dim && M_out->NumCols() == feat_dim);
    M_out->CopyFromMat(Mpart);
  }
  Matrix<double> Mpart_dbl(Mpart);

  // Now have to update the model.
  int32 num_no_data = 0;
  Vector<double> mean(model_dim);
  double tot_occ_means = 0;
  for (int32 p = 0; p < num_pdfs; p++) {
    int32 num_gauss = static_cast<int32>(occs_[p].Dim());
    for (int32 g  = 0; g < num_gauss; g++) {
      double occ = occs_[p](g);
      tot_occ_means += occ;
      if (occ == 0.0) num_no_data++;  // and don't update Gaussian.
      else {
        SubVector<double> mean_stats(mean_accs_[p], g);
        // project mean with transform, to accepted dim.
        mean.AddMatVec(1.0 / occ, Mpart_dbl, kNoTrans, mean_stats, 0.0);
        Vector<BaseFloat> mean_flt(mean);
        am->SetGaussianMean(p, g, mean_flt);
      }
    }
    am->GetPdf(p).ComputeGconsts();
  }
  KALDI_LOG << "Occupancy count used to update means was "
            << tot_occ_means;
  if (num_no_data > 0) {
    KALDI_WARN << num_no_data << " Gaussians not updated due to no data; "
        "be careful not to set your silence-weight to exactly zero (e.g. use 0.01).";
  }
}

}
