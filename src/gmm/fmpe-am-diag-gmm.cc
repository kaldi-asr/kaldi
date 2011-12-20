// gmm/fmpe-am-diag-gmm.cc

// Copyright 2009-2011  Yanmin Qian

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

#include <vector>
#include <set>
#include <algorithm>

#include "gmm/diag-gmm.h"
#include "gmm/fmpe-am-diag-gmm.h"
#include "util/stl-utils.h"
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"

namespace kaldi {

void FmpeAccumModelDiff::Read(std::istream &in_stream, bool binary) {
  int32 dimension, num_components;
  std::string token;

  ExpectMarker(in_stream, binary, "<FMPEMODELDIFFS>");
  ExpectMarker(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectMarker(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);

  Resize(num_components, dimension);

  ReadMarker(in_stream, binary, &token);
  while (token != "</FMPEMODELDIFFS>") {
    if (token == "<MLE_OCCUPANCY>") {
      mle_occupancy_.Read(in_stream, binary);
    } else if (token == "<MEANDIFFS>") {
      mean_diff_accumulator_.Read(in_stream, binary);
    } else if (token == "<DIAGVARDIFFS>") {
      variance_diff_accumulator_.Read(in_stream, binary);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void FmpeAccumModelDiff::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<FMPEMODELDIFFS>");
  WriteMarker(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> occupancy_bf(mle_occupancy_.Dim());
  Matrix<BaseFloat> mean_diff_accumulator_bf(mean_diff_accumulator_.NumRows(),
      mean_diff_accumulator_.NumCols());
  Matrix<BaseFloat> variance_diff_accumulator_bf(variance_diff_accumulator_.NumRows(),
      variance_diff_accumulator_.NumCols());
  occupancy_bf.CopyFromVec(mle_occupancy_);
  mean_diff_accumulator_bf.CopyFromMat(mean_diff_accumulator_);
  variance_diff_accumulator_bf.CopyFromMat(variance_diff_accumulator_);

  WriteMarker(out_stream, binary, "<MLE_OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANDIFFS>");
  mean_diff_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIAGVARDIFFS>");
  variance_diff_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</FMPEMODELDIFFS>");
}

void FmpeAccumModelDiff::Resize(int32 num_comp, int32 dim) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  mle_occupancy_.Resize(num_comp);
  mean_diff_accumulator_.Resize(num_comp, dim);
  variance_diff_accumulator_.Resize(num_comp, dim);
}

void FmpeAccumModelDiff::SetZero() {
  mle_occupancy_.SetZero();
  mean_diff_accumulator_.SetZero();
  variance_diff_accumulator_.SetZero();
}

void FmpeAccumModelDiff::ComputeModelParaDiff(const DiagGmm& diag_gmm,
                                              const AccumDiagGmm& num_acc,
                                              const AccumDiagGmm& den_acc,
                                              const AccumDiagGmm& mle_acc) {
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_); // den_acc.Dim() may not be defined,
  // if we used the "compressed form" of accs where den only has counts.
  KALDI_ASSERT(mle_acc.NumGauss() == num_comp_ && mle_acc.Dim() == dim_);
  
  Matrix<double> mean_diff_tmp(num_comp_, dim_);
  Matrix<double> var_diff_tmp(num_comp_, dim_);
  Matrix<double> mat_tmp(num_comp_, dim_);
  Vector<double> occ_diff(num_comp_);
  Matrix<double> means_invvars(num_comp_, dim_);
  Matrix<double> inv_vars(num_comp_, dim_);

  occ_diff.CopyFromVec(num_acc.occupancy());
  occ_diff.AddVec(-1.0, den_acc.occupancy());

  means_invvars.CopyFromMat(diag_gmm.means_invvars(), kNoTrans);
  inv_vars.CopyFromMat(diag_gmm.inv_vars(), kNoTrans);
  /// compute the means differentials first
  mean_diff_tmp.CopyFromMat(num_acc.mean_accumulator(), kNoTrans);
  if (den_acc.Flags() & kGmmMeans) // probably will be false.
    mean_diff_tmp.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  mean_diff_tmp.MulElements(inv_vars);

  mat_tmp.CopyFromMat(means_invvars, kNoTrans);
  mat_tmp.MulRowsVec(occ_diff);

  mean_diff_tmp.AddMat(-1.0, mat_tmp, kNoTrans);

  /// compute the means differetials
  mean_diff_accumulator_.CopyFromMat(mean_diff_tmp, kNoTrans);

  /// compute the vars differentials second
  var_diff_tmp.CopyFromMat(num_acc.variance_accumulator(), kNoTrans);
  if (den_acc.Flags() & kGmmVariances) // probably will be false.
    var_diff_tmp.AddMat(-1.0, den_acc.variance_accumulator(), kNoTrans);

  var_diff_tmp.MulElements(inv_vars);
  var_diff_tmp.MulElements(inv_vars);
                      
  mat_tmp.CopyFromMat(num_acc.mean_accumulator(), kNoTrans);
  if (den_acc.Flags() & kGmmMeans) // probably will be false.
    mat_tmp.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  mat_tmp.MulElements(inv_vars);
  mat_tmp.MulElements(means_invvars);

  var_diff_tmp.AddMat(-2.0, mat_tmp, kNoTrans);

  mat_tmp.CopyFromMat(means_invvars, kNoTrans);
  mat_tmp.MulElements(means_invvars);
  mat_tmp.AddMat(-1.0, inv_vars, kNoTrans);
  mat_tmp.MulRowsVec(occ_diff);

  var_diff_tmp.AddMat(1.0, mat_tmp, kNoTrans);
  var_diff_tmp.Scale(0.5);

  /// compute the vars differentials
  variance_diff_accumulator_.CopyFromMat(var_diff_tmp, kNoTrans);

  /// copy to obtain the mle occupation probapility
  mle_occupancy_.CopyFromVec(mle_acc.occupancy());
}

void FmpeAccs::Write(std::ostream &out_stream, bool binary) const {
  uint32 tmp_uint32;

  WriteMarker(out_stream, binary, "<FMPEACCS>");

  WriteMarker(out_stream, binary, "<NumGaussians>");
  tmp_uint32 = static_cast<uint32>(config_.gmm_num_comps);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteMarker(out_stream, binary, "<LengthContextExp>");
  tmp_uint32 = static_cast<uint32>(config_.context_windows.NumRows());
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteMarker(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim_);
  if (!binary) out_stream << "\n";

  // convert into BaseFloat before writing things
  Matrix<BaseFloat> mat_bf(dim_, dim_ + 1);

  if (p_.size() != 0) {
    WriteMarker(out_stream, binary, "<P>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.context_windows.NumRows(); ++j) {
		mat_bf.CopyFromMat(p_[i][j], kNoTrans);
        mat_bf.Write(out_stream, binary);
	  }
    }
  }
  if (n_.size() != 0) {
    WriteMarker(out_stream, binary, "<N>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.context_windows.NumRows(); ++j) {
		mat_bf.CopyFromMat(n_[i][j], kNoTrans);
        mat_bf.Write(out_stream, binary);
	  }
    }
  }

  // convert into BaseFloat before writing things
  Vector<BaseFloat> diff_bf(diff_.Dim());
  Vector<BaseFloat> direct_diff_bf(direct_diff_.Dim());
  Vector<BaseFloat> indirect_diff_bf(indirect_diff_.Dim());
  diff_bf.CopyFromVec(diff_);
  direct_diff_bf.CopyFromVec(direct_diff_);
  indirect_diff_bf.CopyFromVec(indirect_diff_);

  WriteMarker(out_stream, binary, "<DIFFERENTIAL>");
  diff_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIRECTDIFFERENTIAL>");
  direct_diff_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<INDIRECTDIFFERENTIAL>");
  indirect_diff_bf.Write(out_stream, binary);

  WriteMarker(out_stream, binary, "</FMPEACCS>");
}

void FmpeAccs::Read(std::istream &in_stream, bool binary,
                         bool add) {
  uint32 tmp_uint32;
  std::string token;

  ExpectMarker(in_stream, binary, "<FMPACCS>");

  ExpectMarker(in_stream, binary, "<NumGaussians>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 num_gaussians = static_cast<int32>(tmp_uint32);
  ExpectMarker(in_stream, binary, "<LengthContExp>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 length_cont_exp = static_cast<int32>(tmp_uint32);
  ExpectMarker(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim_);

  ReadMarker(in_stream, binary, &token);

  while (token != "</FMPEACCS>") {
    if (token == "<P>") {
      p_.resize(num_gaussians);
      for (size_t i = 0; i < p_.size(); ++i) {
        p_[i].resize(length_cont_exp);
		for (size_t j = 0; j < p_[i].size(); ++j) {
          p_[i][j].Read(in_stream, binary, add);
		}
      }
    } else if (token == "<N>") {
      n_.resize(num_gaussians);
      for (size_t i = 0; i < n_.size(); ++i) {
        n_[i].resize(length_cont_exp);
		for (size_t j = 0; j < n_[i].size(); ++j) {
          n_[i][j].Read(in_stream, binary, add);
		}
      }
    } else if (token == "<DIFFERENTIALS>") {
      diff_.Read(in_stream, binary, add);
    } else if (token == "<DIRECTDIFFERENTIALS>") {
      direct_diff_.Read(in_stream, binary, add);
    } else if (token == "<INDIRECTDIFFERENTIALS>") {
      indirect_diff_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void FmpeAccs::ReadModelDiffs(std::istream &in_stream, bool binary) {
  int32 num_pdfs;
  int32 dim;
  ExpectMarker(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim);
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT((num_pdfs > 0) && (dim > 0));

  if (model_diff_accumulators_.size() != static_cast<size_t> (num_pdfs))
    KALDI_ERR << "Reading model differentials but num-pdfs do not match: "
              << (model_diff_accumulators_.size()) << " vs. "
              << (num_pdfs);
  for (std::vector<FmpeAccumModelDiff*>::iterator it = model_diff_accumulators_.begin(),
           end = model_diff_accumulators_.end(); it != end; ++it) {
    (*it)->Read(in_stream, binary);
  }

}

void FmpeAccs::InitPNandDiff(int32 num_gmm_gauss, int32 con_exp, int32 dim) {
    p_.resize(num_gmm_gauss);
    for (int32 i = 0; i < num_gmm_gauss; ++i) {
      p_[i].resize(con_exp);
      for (int32 j = 0; j < con_exp; ++j) {
        p_[i][j].Resize(dim, dim + 1);
      }
    }

    n_.resize(num_gmm_gauss);
    for (int32 i = 0; i < num_gmm_gauss; ++i) {
      n_[i].resize(con_exp);
      for (int32 j = 0; j < con_exp; ++j) {
        n_[i][j].Resize(dim, dim + 1);
      }
    }

	diff_.Resize(dim);
	direct_diff_.Resize(dim);
	indirect_diff_.Resize(dim);
}

void FmpeAccs::InitModelDiff(const AmDiagGmm &model) {
  DeletePointers(&model_diff_accumulators_);  // in case was non-empty when called.
  model_diff_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    model_diff_accumulators_[i] = new FmpeAccumModelDiff();
    model_diff_accumulators_[i]->Resize(model.GetPdf(i));
  }
}

/// Initialization, do InitModelDiff if true when accumulating,
/// and otherwise don't do when sum accumulations
void FmpeAccs::Init(const AmDiagGmm &am_model, bool update) {
  dim_ = am_model.Dim();

  InitPNandDiff(config_.gmm_num_comps, config_.context_windows.NumRows(), dim_);

  if (update) {
	InitModelDiff(am_model);
  }
}

void FmpeAccs::InitializeGMMs(const DiagGmm &gmm, const DiagGmm &gmm_cluster_centers,
                              std::vector<int32> &gaussian_cluster_center_map) {
  gmm_.CopyFromDiagGmm(gmm);
  gmm_cluster_centers_.CopyFromDiagGmm(gmm_cluster_centers);
  gaussian_cluster_center_map_.resize(gaussian_cluster_center_map.size());
  gaussian_cluster_center_map_ = gaussian_cluster_center_map;
}

void FmpeAccs::ComputeOneFrameOffsetFeature(const VectorBase<BaseFloat>& data,
                        std::vector<std::pair<int32, Vector<double> > > *offset) const {
  KALDI_ASSERT((data.Dim() == gmm_.Dim()) && (data.Dim() == gmm_cluster_centers_.Dim()));
  KALDI_ASSERT((gmm_.NumGauss() != 0) && (gmm_cluster_centers_.NumGauss() != 0)
               && (gmm_.NumGauss() > gmm_cluster_centers_.NumGauss())
               && (config_.gmm_cluster_centers_nbest < gmm_cluster_centers_.NumGauss())
               && (config_.gmm_gaussian_nbest < gmm_.NumGauss()))

  int32 dim = data.Dim();
  int32 num_gauss = gmm_.NumGauss();
  int32 num_cluster_centers = gmm_cluster_centers_.NumGauss();
  int32 gmm_cluster_centers_nbest = config_.gmm_cluster_centers_nbest;

  std::set<int32> pruned_centers;
  Vector<BaseFloat> loglikes(num_cluster_centers);
  gmm_cluster_centers_.LogLikelihoods(data, &loglikes);
  Vector<BaseFloat> loglikes_copy(loglikes);
  BaseFloat *ptr = loglikes_copy.Data();
  std::nth_element(ptr, ptr+num_cluster_centers-gmm_cluster_centers_nbest, ptr+num_cluster_centers);
  BaseFloat thresh = ptr[num_cluster_centers-gmm_cluster_centers_nbest];
  for (int32 g = 0; g < num_cluster_centers; g++) {
    if (loglikes(g) >= thresh)
      pruned_centers.insert(g);
  }

  std::vector< std::pair<double, int32> > pruned_gauss;
  for (int32 gauss_index = 0; gauss_index < num_gauss; gauss_index++) {
    int32 current_cluster = gaussian_cluster_center_map_[gauss_index];
    if (pruned_centers.end() != pruned_centers.find(current_cluster)) {
      double loglike = gmm_.ComponentLogLikelihood(data, gauss_index);
      pruned_gauss.push_back(std::make_pair(loglike, gauss_index));
    }
  }
  KALDI_ASSERT(!pruned_gauss.empty());

  int32 gmm_gaussian_nbest = config_.gmm_gaussian_nbest;
  std::nth_element(pruned_gauss.begin(),
                   pruned_gauss.end() - gmm_gaussian_nbest,
                   pruned_gauss.end());
  pruned_gauss.erase(pruned_gauss.begin(),
                     pruned_gauss.end() - gmm_gaussian_nbest);

  double weight = 0.0;
  for (int32 i = 0; i < pruned_gauss.size(); ++i) {
    weight += exp(pruned_gauss[i].first);
  }
  for (int32 i = 0; i < pruned_gauss.size(); ++i) {
    pruned_gauss[i].first = exp(pruned_gauss[i].first) / weight;
  }

  Vector<BaseFloat> tmp_offset(dim + 1);
  SubVector<BaseFloat> sub_tmp_offset(tmp_offset, 1, dim);
  Vector<BaseFloat> tmp_mean(dim);
  Vector<BaseFloat> tmp_var(dim);
  for (int32 i = 0; i < pruned_gauss.size(); ++i) {
	tmp_offset(0) = pruned_gauss[i].first * 5.0;
    sub_tmp_offset.CopyFromVec(data);
    gmm_.GetComponentMean(pruned_gauss[i].second, &tmp_mean);
    sub_tmp_offset.AddVec(-1.0, tmp_mean);
    gmm_.GetComponentVariance(pruned_gauss[i].second, &tmp_var);
    tmp_var.ApplyPow(0.5);
    sub_tmp_offset.DivElemByElem(tmp_var);
    sub_tmp_offset.Scale(pruned_gauss[i].first);

    offset->push_back(std::make_pair(pruned_gauss[i].second, tmp_offset));
  }
}

void FmpeAccs::ComputeWholeFileOffsetFeature(const MatrixBase<BaseFloat>& data,
           std::vector<std::vector<std::pair<int32, Vector<double> > > > *whole_file_offset) const {
  int32 nframe = data.NumRows();
  whole_file_offset->reserve(nframe);

  for (int32 i = 0; i < nframe; i++) {
	std::vector<std::pair<int32, Vector<double> > > offset;
    ComputeOneFrameOffsetFeature(data.Row(i), &offset);
    whole_file_offset->push_back(offset);
  }
}

bool Gauss_index_lower(std::pair<int32, Vector<double> > M,
					   std::pair<int32, Vector<double> >  N) {
  return M.first < N.first;
}

void FmpeAccs::ComputeContExpOffsetFeature(
       const std::vector<std::vector<std::pair<int32, Vector<double> > >* > &offset_win,
       std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > *ht) const {
  KALDI_ASSERT((config_.context_windows.NumCols() == offset_win.size()));

  std::vector<std::pair<int32, Vector<double> > > offset_tmp;
  std::vector<std::pair<int32, Vector<double> > > offset_uniq_tmp;

  for (int32 i = 0; i < config_.context_windows.NumRows(); i++) {
	// for every context
	for (int32 j = 0; j < config_.context_windows.NumCols(); j++) {
	  if (config_.context_windows(i, j) > 0.0) {
		if (offset_win[j]->empty() == 0) {
		  for (int32 k = 0; k < offset_win[j]->size(); k++) {
	        offset_tmp.push_back((*offset_win[j])[k]);
	        offset_tmp.back().second.Scale(config_.context_windows(i, j));
	      }
		}
	  }
	}

	if (offset_tmp.empty() == 0) {
	  std::sort(offset_tmp.begin(), offset_tmp.end(), Gauss_index_lower);
	  offset_uniq_tmp.push_back(offset_tmp[0]);
	  for (int32 igauss = 1; igauss < offset_tmp.size(); igauss++) {
	    if (offset_tmp[igauss].first == offset_tmp[igauss - 1].first) {
		  offset_uniq_tmp.back().second.AddVec(1.0, offset_tmp[igauss].second);
	    } else {
          offset_uniq_tmp.push_back(offset_tmp[igauss]);
	    }
	  }

	  ht->push_back(std::make_pair(i, offset_uniq_tmp));
      offset_tmp.clear();
	  offset_uniq_tmp.clear();
	}
  }
}

void FmpeAccs::ComputeHighDimemsionFeature(
     const std::vector<std::vector<std::pair<int32, Vector<double> > > > &whole_file_offset_feat,
	 int32 frame_index,
     std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > *ht) const {
  KALDI_ASSERT((frame_index >= 0) && (frame_index < whole_file_offset_feat.size()));

  int32 lenght_context_windows = config_.context_windows.NumCols();
  int32 half_len_win = lenght_context_windows / 2;
  int32 num_frame = whole_file_offset_feat.size();
  std::vector<std::vector<std::pair<int32, Vector<double> > >* > offset_win;
  std::vector<std::pair<int32, Vector<double> > > empty_feat;

  for (int32 i = (frame_index - half_len_win);
	   i < (frame_index - half_len_win + lenght_context_windows); i++) {
	/// we append zero if the index is out of the whole file feature lenght
	if ((i < 0) || (i >= num_frame)) {
	  offset_win.push_back(&empty_feat);
	} else {
	  offset_win.push_back(
                 const_cast<std::vector<std::pair<int32, Vector<double> > >* >
				 (&(whole_file_offset_feat[i])));
	}
  }

  ComputeContExpOffsetFeature(offset_win, ht);
}

void FmpeAccs::ProjectHighDimensionFeature(
         const std::vector< std::vector< Matrix<double> > > &M,
         const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht,
         Vector<double> *fea_out) const {
  KALDI_ASSERT((M.size() == gmm_.NumGauss())
			   && (M[0].size() == ht.size())
			   && (M[0][0].NumRows() == gmm_.Dim())
			   && (M[0][0].NumCols() == gmm_.Dim() + 1));

  int32 dim = gmm_.Dim();
  Vector<double> tmp_fea(dim);
  tmp_fea.SetZero();

  for(int32 i = 0; i < ht.size(); i++) {
	int32 cont_index = ht[i].first;
	for (int32 j = 0; j < ht[i].second.size(); j++) {
      int32 gauss_index = ht[i].second[j].first;
	  tmp_fea.AddMatVec(1.0, M[gauss_index][cont_index], kNoTrans, ht[i].second[j].second, 1.0);
	}
  }

  fea_out->CopyFromVec(tmp_fea);
}

void FmpeAccs::ObtainNewFmpeFeature(
    const VectorBase<BaseFloat> &data,
    const std::vector< std::vector< Matrix<double> > > &M,
    const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht,
    Vector<double> *fea_new) const {
  KALDI_ASSERT((data.Dim() == gmm_.Dim()));

  Vector<double> tmp_fea(data.Dim());
  ProjectHighDimensionFeature(M, ht, &tmp_fea);

  fea_new->CopyFromVec(data);
  fea_new->AddVec(1.0, tmp_fea);
}

void FmpeAccs::AccumulateDirectDiffFromPosteriors(const DiagGmm &gmm,
                                            const VectorBase<BaseFloat> &data,
											const VectorBase<BaseFloat> &posteriors,
                                            Vector<double> *direct_diff) {
  KALDI_ASSERT(gmm.Dim() == Dim());
  KALDI_ASSERT(gmm.NumGauss() == posteriors.Dim());
  KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());
  KALDI_ASSERT(direct_diff->Dim() == Dim());

  Matrix<double> means_invvars(gmm.NumGauss(), gmm.Dim());
  Matrix<double> inv_vars(gmm.NumGauss(), gmm.Dim());
  Matrix<double> data_tmp(gmm.NumGauss(), gmm.Dim());
  Matrix<double> mat_tmp(gmm.NumGauss(), gmm.Dim());
  Vector<double> post_scale(gmm.NumGauss());

  means_invvars.CopyFromMat(gmm.means_invvars(), kNoTrans);
  inv_vars.CopyFromMat(gmm.inv_vars(), kNoTrans);

  for (int32 i = 0; i < data_tmp.NumRows(); i++) {
	data_tmp.Row(i).AddVec(1.0, data);
  }
  data_tmp.MulElements(inv_vars);

  mat_tmp.CopyFromMat(means_invvars, kNoTrans);
  mat_tmp.AddMat(-1.0, data_tmp, kNoTrans);

  post_scale.CopyFromVec(posteriors);
  post_scale.Scale(config_.lat_prob_scale);

  direct_diff->AddMatVec(1.0, mat_tmp, kTrans, post_scale, 1.0);
}

void FmpeAccs::AccumulateInDirectDiffFromPosteriors(const DiagGmm &gmm,
                             const FmpeAccumModelDiff &fmpe_diaggmm_diff_acc,
                             const VectorBase<BaseFloat> &data,
                             const VectorBase<BaseFloat> &posteriors,
							 Vector<double> *indirect_diff) {
  KALDI_ASSERT(gmm.NumGauss() == fmpe_diaggmm_diff_acc.NumGauss());
  KALDI_ASSERT(gmm.NumGauss() == posteriors.Dim());
  KALDI_ASSERT(gmm.Dim() == fmpe_diaggmm_diff_acc.Dim());
  KALDI_ASSERT(gmm.Dim() == Dim());
  KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());
  KALDI_ASSERT(indirect_diff->Dim() == Dim());

  Matrix<double> mat_tmp(gmm.NumGauss(), gmm.Dim());
  Vector<double> vec_tmp(gmm.NumGauss());

  gmm.GetMeans(&mat_tmp);
  for (int32 i = 0; i < mat_tmp.NumRows(); i++) {
	mat_tmp.Row(i).AddVec(-1.0, data);
  }
  mat_tmp.MulElements(fmpe_diaggmm_diff_acc.variance_diff_accumulator());
  mat_tmp.Scale(-2.0);
  mat_tmp.AddMat(1.0, fmpe_diaggmm_diff_acc.mean_diff_accumulator(), kNoTrans);
  // should be scaled in compute model difficientials,
  // but used here just for convenient
  mat_tmp.Scale(config_.lat_prob_scale);

  vec_tmp.CopyFromVec(posteriors);
  vec_tmp.DivElemByElem(fmpe_diaggmm_diff_acc.mle_occupancy());

  indirect_diff->AddMatVec(1.0, mat_tmp, kTrans, vec_tmp, 1.0);
}

void FmpeAccs::AccumulateInDirectDiffFromDiag(const DiagGmm &gmm,
                             const FmpeAccumModelDiff &fmpe_diaggmm_diff_acc,
                             const VectorBase<BaseFloat> &data,
                             BaseFloat frame_posterior,
							 Vector<double> *indirect_diff) {
  KALDI_ASSERT(gmm.NumGauss() == fmpe_diaggmm_diff_acc.NumGauss());
  KALDI_ASSERT(gmm.Dim() == fmpe_diaggmm_diff_acc.Dim());
  KALDI_ASSERT(gmm.Dim() == Dim());
  KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());
  KALDI_ASSERT(indirect_diff->Dim() == Dim());

  Vector<BaseFloat> posteriors(gmm.NumGauss());
  gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

  AccumulateInDirectDiffFromPosteriors(gmm, fmpe_diaggmm_diff_acc,
									   data, posteriors, indirect_diff);
}

void FmpeAccs::AccumulateFromDifferential(const VectorBase<double> &direct_diff,
										  const VectorBase<double> &indirect_diff,
       const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht) {
  KALDI_ASSERT((direct_diff.Dim() == indirect_diff.Dim()));
  KALDI_ASSERT(direct_diff.Dim() == Dim());

  Vector<double> diff(direct_diff);
  diff.AddVec(1.0, indirect_diff);

  int32 dim = gmm_.Dim();
  Matrix<double> tmp(dim, dim + 1);
  tmp.SetZero();

  /// accumulate the p and n statistics
  for (int32 i = 0; i < ht.size(); i++) {
	int32 cont_index = ht[i].first;
	for (int32 j = 0; j < ht[i].second.size(); j++) {
      int32 gauss_index = ht[i].second[j].first;
	  tmp.AddVecVec(1.0, diff, ht[i].second[j].second);

      for (int32 r = 0; r < dim; r++) {
        for (int32 c = 0;c < (dim + 1); c++) {
			if (tmp(r, c) > 0.0) {
		      p_[gauss_index][cont_index](r, c) += tmp(r, c);
			}
	        else {
		      n_[gauss_index][cont_index](r, c) -= tmp(r, c);
			}
		}
	  }

	  tmp.SetZero();
	}
  }

  /// accumulate the direct/indirect and total differentials
  diff_.AddVec(1.0, diff);
  direct_diff_.AddVec(1.0, direct_diff);
  indirect_diff_.AddVec(1.0, indirect_diff);
}

FmpeUpdater::FmpeUpdater(const FmpeAccs &accs)
      : config_(accs.config()), dim_(accs.Dim()) {
  Init(config_.gmm_num_comps, config_.context_windows.NumRows(), dim_);
};

FmpeUpdater::FmpeUpdater(const FmpeUpdater &other)
	: config_(other.config_), avg_std_var_(other.avg_std_var_),
	  dim_(other.dim_) {
  if (other.M_.size() != 0) {
    M_.resize(other.M_.size());
    for (int32 i = 0; i < other.M_.size(); ++i) {
      M_[i].resize(other.M_[i].size());
      for (int32 j = 0; j < other.M_[i].size(); ++j) {
        M_[i][j].Resize(other.M_[i][j].NumRows(), other.M_[i][j].NumCols());
        M_[i][j].CopyFromMat(other.M_[i][j], kNoTrans);
      }
    }
  }
}

void FmpeUpdater::Init(int32 num_gmm_gauss, int32 con_exp, int32 dim) {
    M_.resize(num_gmm_gauss);
    for (int32 i = 0; i < num_gmm_gauss; ++i) {
      M_[i].resize(con_exp);
      for (int32 j = 0; j < con_exp; ++j) {
        M_[i][j].Resize(dim, dim + 1);
      }
    }

	avg_std_var_.Resize(dim);
}

void FmpeUpdater::Write(std::ostream &out_stream, bool binary) const {
  uint32 tmp_uint32;

  WriteMarker(out_stream, binary, "<FMPE>");

  WriteMarker(out_stream, binary, "<NumGaussians>");
  tmp_uint32 = static_cast<uint32>(config_.gmm_num_comps);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteMarker(out_stream, binary, "<LengthContExp>");
  tmp_uint32 = static_cast<uint32>(config_.context_windows.NumRows());
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteMarker(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim_);
  if (!binary) out_stream << "\n";

  // convert into BaseFloat before writing things
  Matrix<BaseFloat> mat_bf(dim_, dim_ + 1);

  if (M_.size() != 0) {
    WriteMarker(out_stream, binary, "<PROJ_MAT>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.context_windows.NumRows(); ++j) {
		mat_bf.CopyFromMat(M_[i][j], kNoTrans);
        mat_bf.Write(out_stream, binary);
	  }
    }
  }

  WriteMarker(out_stream, binary, "</FMPE>");
}

void FmpeUpdater::Read(std::istream &in_stream, bool binary) {
  uint32 tmp_uint32;
  std::string token;

  ExpectMarker(in_stream, binary, "<FMPE>");

  ExpectMarker(in_stream, binary, "<NumGaussians>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 num_gaussians = static_cast<int32>(tmp_uint32);
  ExpectMarker(in_stream, binary, "<LengthContExp>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 length_cont_exp = static_cast<int32>(tmp_uint32);
  ExpectMarker(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim_);

  ReadMarker(in_stream, binary, &token);

  while (token != "</FMPE>") {
    if (token == "<PROJ_MAT>") {
      M_.resize(num_gaussians);
      for (size_t i = 0; i < M_.size(); ++i) {
        M_[i].resize(length_cont_exp);
		for (size_t j = 0; j < M_[i].size(); ++j) {
          M_[i][j].Read(in_stream, binary);
		}
      }
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void FmpeUpdater::ComputeAvgStandardDeviation(const AmDiagGmm &am) {
  Matrix<double> vars_tmp;
  Vector<double> vec_tmp(am.Dim());

  for (int32 i = 0; i < am.NumPdfs(); i++) {
	const DiagGmm &gmm = am.GetPdf(i);
	gmm.GetVars(&vars_tmp);
	vars_tmp.ApplyPow(0.5);
	vec_tmp.AddRowSumMat(vars_tmp);
  }

  vec_tmp.Scale(1 / am.NumGauss());

  avg_std_var_.CopyFromVec(vec_tmp);
}

void FmpeUpdater::Update(const FmpeAccs &accs,
					     BaseFloat *obj_change_out,
                         BaseFloat *count_out) {
  KALDI_ASSERT((M_.size() == accs.pos().size()) && (M_.size() == accs.neg().size()));
  KALDI_ASSERT((M_[0].size() == accs.pos()[0].size()) && (M_[0].size() == accs.neg()[0].size())
			   && M_[0].size() == config_.context_windows.NumRows());
  KALDI_ASSERT((M_[0][0].NumRows() == accs.pos()[0][0].NumRows())
			   && (M_[0][0].NumRows() == accs.neg()[0][0].NumRows())
			   && (M_[0][0].NumRows() == avg_std_var_.Dim()));
  KALDI_ASSERT((M_[0][0].NumCols() == accs.pos()[0][0].NumCols())
			   && (M_[0][0].NumCols() == accs.neg()[0][0].NumCols())
			   && (M_[0][0].NumCols() == (M_[0][0].NumRows() + 1)));

  int32 ngauss = M_.size();
  int32 n_cont_exp = M_[0].size();
  int32 dim = M_[0][0].NumRows();

  Matrix<double> pandn_add_tmp(dim, dim + 1);
  Matrix<double> pandn_sub_tmp(dim, dim + 1);
  Vector<double> vec_tmp(avg_std_var_);
  vec_tmp.Scale(1 / config_.E);

  KALDI_LOG << "Updating the projection matrix M, the dim is: [ "
	        << ngauss << " ][ " << n_cont_exp << " ][ " << dim << " ][ " << dim + 1
			<< " ] -> [nGauss][nContExp][fea_dim][fea_dim + 1]";

  for (int32 gauss_index = 0; gauss_index < ngauss; gauss_index++) {
	for (int32 icon_exp = 0; icon_exp < n_cont_exp; icon_exp++) {
		pandn_add_tmp.CopyFromMat(accs.pos()[gauss_index][icon_exp], kNoTrans);
		pandn_add_tmp.AddMat(1.0, accs.neg()[gauss_index][icon_exp], kNoTrans);
		pandn_sub_tmp.CopyFromMat(accs.pos()[gauss_index][icon_exp], kNoTrans);
		pandn_sub_tmp.AddMat(-1.0, accs.neg()[gauss_index][icon_exp], kNoTrans);
		pandn_sub_tmp.DivElements(pandn_add_tmp);
		pandn_sub_tmp.MulRowsVec(vec_tmp);

		M_[gauss_index][icon_exp].AddMat(1.0, pandn_sub_tmp, kNoTrans);
	}
  }

  /// add some code to calculate the objective function change // TODO
}

void ClusterGmmToClusterCenters(const DiagGmm &gmm,
                                int32 num_cluster_centers,
                                BaseFloat cluster_varfloor,
                                DiagGmm *ubm_cluster_centers,
                                std::vector<int32> *cluster_center_map) {
  // Bottom-up clustering of the Gaussians in the gmm model
  KALDI_ASSERT(num_cluster_centers < gmm.NumGauss());
  int32 dim = gmm.Dim();
  Vector<BaseFloat> tmp_mean(dim);
  Vector<BaseFloat> tmp_var(dim);
  int32 num_gaussians = gmm.NumGauss();
  std::vector<Clusterable*> gauss_clusters;
  gauss_clusters.reserve(num_cluster_centers);

  for (int32 gauss_index = 0; gauss_index < num_gaussians; gauss_index++) {
    gmm.GetComponentMean(gauss_index, &tmp_mean);
    gmm.GetComponentVariance(gauss_index, &tmp_var);
    tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
    BaseFloat this_weight = gmm.weights()(gauss_index);
    tmp_mean.Scale(this_weight);
    tmp_var.Scale(this_weight);
    gauss_clusters.push_back(new GaussClusterable(tmp_mean, tmp_var,
                          cluster_varfloor, this_weight));
  }

  std::vector<Clusterable*> gauss_clusters_out;
  KALDI_VLOG(1) << "Creating " << num_cluster_centers << " gaussian clusters centers.";
  ClusterBottomUp(gauss_clusters, kBaseFloatMax, num_cluster_centers,
                  &gauss_clusters_out,
                  cluster_center_map /*get the cluster assignments*/);
  DeletePointers(&gauss_clusters);

  // Next, put the clustered Gaussians centers into a single GMM.
  KALDI_VLOG(1) << "Putting " << num_cluster_centers << " Gaussians cluster centers"
                << "into a single GMM model.";
  Matrix<BaseFloat> tmp_means(num_cluster_centers, dim);
  Matrix<BaseFloat> tmp_vars(num_cluster_centers, dim);
  Vector<BaseFloat> tmp_weights(num_cluster_centers);
  Vector<BaseFloat> tmp_vec(dim);
  DiagGmm tmp_gmm;
  for (int32 gauss_index = 0; gauss_index < num_cluster_centers; gauss_index++) {
    GaussClusterable *this_cluster = static_cast<GaussClusterable*>(
        gauss_clusters_out[gauss_index]);
    BaseFloat weight = this_cluster->count();
    tmp_weights(gauss_index) = weight;
    tmp_vec.CopyFromVec(this_cluster->x_stats());
    tmp_vec.Scale(1/weight);
    tmp_means.CopyRowFromVec(tmp_vec, gauss_index);
    tmp_vec.CopyFromVec(this_cluster->x2_stats());
    tmp_vec.Scale(1/weight);
    tmp_vec.AddVec2(-1.0, tmp_means.Row(gauss_index));  // x^2 stats to var.
    tmp_vars.CopyRowFromVec(tmp_vec, gauss_index);
  }
  DeletePointers(&gauss_clusters_out);

  tmp_gmm.Resize(num_cluster_centers, dim);
  tmp_weights.Scale(1.0/tmp_weights.Sum());
  tmp_gmm.SetWeights(tmp_weights);
  tmp_vars.InvertElements();  // need inverse vars...
  tmp_gmm.SetInvVarsAndMeans(tmp_vars, tmp_means);

  KALDI_VLOG(1) << "Obtain " << tmp_gmm.NumGauss() << " Gaussians cluster centers.";
  ubm_cluster_centers->CopyFromDiagGmm(tmp_gmm);
}

void ObtainUbmAndSomeClusterCenters(
			         const AmDiagGmm &am,
                     const Vector<BaseFloat> &state_occs,
                     const FmpeConfig &config,
                     DiagGmm *gmm_out,
                     DiagGmm *gmm_cluster_centers_out,
                     std::vector<int32> *gaussian_cluster_center_map_out) {
  /// First clusters the Gaussians in an acoustic model to a single GMM with specified
  /// number of components. Using the same algorithm in the SGMM's UBM
  /// initialization
  kaldi::UbmClusteringOptions ubm_opts;
  ubm_opts.ubm_numcomps = config.gmm_num_comps;
  ClusterGaussiansToUbm(am, state_occs, ubm_opts, gmm_out);

  /// Clusters the Gaussians in the gmm model to some cluster centers, which is for
  /// more efficient evaluation of the gaussian posteriors just with
  /// the most likely cluster centers
  ClusterGmmToClusterCenters(*gmm_out, config.gmm_num_cluster_centers, config.cluster_varfloor,
                             gmm_cluster_centers_out, gaussian_cluster_center_map_out);

}

}  // End of namespace kaldi
