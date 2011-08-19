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
#include <map>

#include "gmm/diag-gmm.h"
#include "gmm/fmpe-am-diag-gmm.h"
#include "util/stl-utils.h"
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"

namespace kaldi {

void FmpeAccumModelDiff::Resize(int32 num_comp, int32 dim) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  occupancy_.Resize(num_comp);
  mean_diff_accumulator_.Resize(num_comp, dim);
  variance_diff_accumulator_.Resize(num_comp, dim);
}

void FmpeAccumModelDiff::SetZero() {
  occupancy_.SetZero();
  mean_diff_accumulator_.SetZero();
  variance_diff_accumulator_.SetZero();
}

void FmpeAccumModelDiff::AccumulateFromMpeStats(const DiagGmm& diag_gmm,
							const AccumDiagGmm& num_acc,
                            const AccumDiagGmm& den_acc,
							const AccumDiagGmm& mle_acc,
                            const MmieDiagGmmOptions& opts) {
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_ && den_acc.Dim() == dim_);
  KALDI_ASSERT(mle_acc.NumGauss() == num_comp_ && mle_acc.Dim() == dim_);

  Matrix<double> mean_diff_tmp(num_comp_, dim_);
  Matrix<double> means_tmp(num_comp_, dim_);
  Matrix<double> vars_tmp(num_comp_, dim_);
  Vector<double> occ_tmp(num_comp_);

  /// compute the means differentials first
  diag_gmm.GetMeans(&means_tmp);
  occ_tmp.CopyFromVec(num_acc.occupancy());
  occ_tmp.AddVec(-1.0, den_acc.occupancy());
  means_tmp.MulRowsVec(occ_tmp);
  mean_diff_tmp.CopyFromMat(num_acc.mean_accumulator(), kNoTrans);
  mean_diff_tmp.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  mean_diff_tmp.AddMat(-1.0, means_tmp, kNoTrans);
  diag_gmm.GetVars(&vars_tmp);
  means_tmp.DivElements(vars_tmp);

  /// compute the means differetials
  mean_diff_accumulator_.CopyFromMat(means_tmp, kNoTrans);

  /// compute the vars differentials second
  /// compute the variance of the num/den statistics around the current mean
  Matrix<double> svars_means_num(num_comp_, dim_);
  Matrix<double> svars_means_den(num_comp_, dim_);
  Matrix<double> mat_tmp(num_comp_, dim_);
  Vector<double> vec_tmp(num_comp_);

  /// the variance around the num stats
  svars_means_num.CopyFromMat(num_acc.variance_accumulator(), kNoTrans);
  diag_gmm.GetMeans(&mat_tmp);
  mat_tmp.MulElements(num_acc.mean_accumulator());
  mat_tmp.Scale(2.0);
  svars_means_num.AddMat(-1.0, mat_tmp, kNoTrans);
  diag_gmm.GetMeans(&mat_tmp);
  mat_tmp.ApplyPow(2.0);
  mat_tmp.MulRowsVec(num_acc.occupancy());
  svars_means_num.AddMat(1.0, mat_tmp, kNoTrans);
  vec_tmp.CopyFromVec(num_acc.occupancy());
  vec_tmp.InvertElements();
  svars_means_num.MulRowsVec(vec_tmp);

  /// the variance around the den stats
  svars_means_den.CopyFromMat(den_acc.variance_accumulator(), kNoTrans);
  diag_gmm.GetMeans(&mat_tmp);
  mat_tmp.MulElements(den_acc.mean_accumulator());
  mat_tmp.Scale(2.0);
  svars_means_den.AddMat(-1.0, mat_tmp, kNoTrans);
  diag_gmm.GetMeans(&mat_tmp);
  mat_tmp.ApplyPow(2.0);
  mat_tmp.MulRowsVec(den_acc.occupancy());
  svars_means_den.AddMat(1.0, mat_tmp, kNoTrans);
  vec_tmp.CopyFromVec(den_acc.occupancy());
  vec_tmp.InvertElements();
  svars_means_den.MulRowsVec(vec_tmp);

  /// compute the vars differentials
  diag_gmm.GetVars(&mat_tmp);
  mat_tmp.InvertElements();

  svars_means_num.MulElements(mat_tmp);
  svars_means_num.Add(-1.0);
  svars_means_num.MulElements(mat_tmp);

  svars_means_den.MulElements(mat_tmp);
  svars_means_den.Add(-1.0);
  svars_means_den.MulElements(mat_tmp);

  vec_tmp.CopyFromVec(num_acc.occupancy());
  svars_means_num.MulRowsVec(vec_tmp);
  svars_means_num.Scale(0.5);

  vec_tmp.CopyFromVec(den_acc.occupancy());
  svars_means_den.MulRowsVec(vec_tmp);
  svars_means_den.Scale(0.5);

  variance_diff_accumulator_.CopyFromMat(svars_means_num, kNoTrans);
  variance_diff_accumulator_.AddMat(-1.0, svars_means_den, kNoTrans);

  /// copy to obtain the mle occupation probapility
  occupancy_.CopyFromVec(mle_acc.occupancy());
}

void FmpeAccs::Write(std::ostream &out_stream, bool binary) const {
  uint32 tmp_uint32;

  WriteMarker(out_stream, binary, "<FMPEACCS>");

  WriteMarker(out_stream, binary, "<NumGaussians>");
  tmp_uint32 = static_cast<uint32>(config_.gmm_num_comps);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteMarker(out_stream, binary, "<LengthContextExp>");
  tmp_uint32 = static_cast<uint32>(config_.nlength_context_expansion);
  WriteBasicType(out_stream, binary, tmp_uint32);
  if (!binary) out_stream << "\n";

  if (p_.size() != 0) {
    WriteMarker(out_stream, binary, "<P>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.nlength_context_expansion; ++j) {
        p_[i][j].Write(out_stream, binary);
	  }
    }
  }
  if (n_.size() != 0) {
    WriteMarker(out_stream, binary, "<N>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.nlength_context_expansion; ++j) {
        n_[i][j].Write(out_stream, binary);
	  }
    }
  }

  WriteMarker(out_stream, binary, "<DIFFERENTIAL>");
  diff_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIRECTDIFFERENTIAL>");
  direct_diff_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<INDIRECTDIFFERENTIAL>");
  indirect_diff_.Write(out_stream, binary);

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

  InitPNandDiff(config_.gmm_num_comps, config_.nlength_context_expansion, dim_);

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


void FmpeAccs::ComputeContExpOffsetFeature(
                const std::vector<std::vector<std::pair<int32, Vector<double> > > > &offset_win,
                const Vector<double> &frame_weight,
                std::map<int32, std::vector<std::pair<int32, Vector<double> > > > *ht) const {
  std::vector<std::pair<int32, Vector<double> > > offset_tmp;
  int32 icontexp = 0;
  int32 iframe = 0;

  /// for the context feature in position -4 (0)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context -4 (0)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position -3 (1)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context -3 (1)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position -2 (2)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context -2 (2)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position -1 (3)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context -1 (3)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();


  /// for the context feature in position 0 (4)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context 0 (4)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position 1 (5)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context 1 (5)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position 2 (6)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context 2 (6)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position 3 (7)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context 3 (7)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

  /// for the context feature in position 4 (8)
  offset_tmp = offset_win[iframe];
  for (int32 i = 0; i < offset_tmp.size(); i++) {
    offset_tmp[i].second.Scale(frame_weight(iframe));
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;
  for (int32 i = 0; i < offset_win[iframe].size(); i++) {
	int32 flag = 0;
	for (int32 j = 0; j < offset_tmp.size(); j++) {
	  if (offset_tmp[j].first == offset_win[iframe][i].first) {
		offset_tmp[j].second.AddVec(frame_weight(iframe), offset_win[iframe][i].second);
		flag = 1;
		break;
	  }
	}
	if (flag == 0) {  // not exist about this gaussian's offset
	  offset_tmp.push_back(offset_win[iframe][i]);
	  offset_tmp.back().second.Scale(frame_weight(iframe));
	}
  }
  iframe++;

  /// insert in the final hi dimension feature vector for context 4 (8)
  ht->insert(std::make_pair(icontexp, offset_tmp));
  icontexp++;
  offset_tmp.clear();

}

void FmpeAccs::ProjectHighDimensionFeature(
         const std::vector< std::vector< Matrix<double> > > &M,
         const std::map<int32, std::vector<std::pair<int32, Vector<double> > > > &ht,
         Vector<double> *fea_out) const {
  KALDI_ASSERT((M.size() == gmm_.NumGauss())
			   && (M[0].size() == ht.size())
			   && (M[0][0].NumRows() == gmm_.Dim())
			   && (M[0][0].NumCols() == gmm_.Dim() + 1));

  int32 dim = gmm_.Dim();
  std::map<int32, std::vector<std::pair<int32, Vector<double> > > >::const_iterator ht_Iter;
  Vector<double> tmp_fea(dim);
  tmp_fea.SetZero();

  for (ht_Iter = ht.begin(); ht_Iter != ht.end(); ht_Iter++) {
	int32 cont_index = ht_Iter->first;
	for (int32 i = 0; i < ht_Iter->second.size(); i++) {
      int32 gauss_index = ht_Iter->second[i].first;
	  tmp_fea.AddMatVec(1.0, M[gauss_index][cont_index], kNoTrans, ht_Iter->second[i].second, 1.0);
	}
  }

  fea_out->CopyFromVec(tmp_fea);
}

void FmpeAccs::ObtainNewFmpeFeature(const VectorBase<BaseFloat> &data,
         const std::vector< std::vector< Matrix<double> > > &M,
         const std::map<int32, std::vector<std::pair<int32, Vector<double> > > > &ht,
         Vector<double> *fea_new) const {
  KALDI_ASSERT((data.Dim() == gmm_.Dim()));

  Vector<double> tmp_fea(data.Dim());
  ProjectHighDimensionFeature(M, ht, &tmp_fea);

  fea_new->CopyFromVec(data);
  fea_new->AddVec(1.0, tmp_fea);
}

void FmpeAccs::AccumulateDirectDiffFromDiag(const DiagGmm &gmm,
                             const VectorBase<BaseFloat> &data,
                             BaseFloat frame_posterior,
							 Vector<double> *direct_diff) {
  assert(gmm.Dim() == Dim());
  assert(static_cast<int32>(data.Dim()) == Dim());

  Vector<BaseFloat> posteriors(gmm.NumGauss());
  /// we should need to use the scaled phone arc's
  /// gaussian occupation probability here. // TODO
//  BaseFloat log_like = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

//  AccumulateFromPosteriors(data, posteriors);
  Matrix<double> means_tmp(gmm.NumGauss(), gmm.Dim());
  Matrix<double> vars_tmp(gmm.NumGauss(), gmm.Dim());
  Vector<double> vec_tmp(gmm.NumGauss());

  gmm.GetMeans(&means_tmp);
  gmm.GetVars(&vars_tmp);
  for (int32 i = 0; i < means_tmp.NumRows(); i++) {
	means_tmp.Row(i).AddVec(-1.0, data);
  }
  means_tmp.DivElements(vars_tmp);

  vec_tmp.CopyFromVec(posteriors);
  vec_tmp.Scale(config_.lat_prob_scale);

  direct_diff->Resize(data.Dim());
  direct_diff->AddMatVec(1.0, means_tmp, kTrans, vec_tmp, 0.0);
}

void FmpeAccs::AccumulateInDirectDiffFromDiag(const DiagGmm &gmm,
                             const FmpeAccumModelDiff fmpe_diaggmm_diff_acc,
                             const VectorBase<BaseFloat> &data,
                             BaseFloat frame_posterior,
							 Vector<double> *indirect_diff) {
  assert(gmm.NumGauss() == fmpe_diaggmm_diff_acc.NumGauss());
  assert(gmm.Dim() == fmpe_diaggmm_diff_acc.Dim());
  assert(gmm.Dim() == Dim());
  assert(static_cast<int32>(data.Dim()) == Dim());

  Vector<BaseFloat> posteriors(gmm.NumGauss());
  gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

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
  vec_tmp.DivElemByElem(fmpe_diaggmm_diff_acc.occupancy());

  indirect_diff->Resize(data.Dim());
  indirect_diff->AddMatVec(1.0, mat_tmp, kTrans, vec_tmp, 0.0);
}

void FmpeAccs::AccumulateFromDifferential(const VectorBase<double> &direct_diff,
										  const VectorBase<double> &indirect_diff,
       const std::map<int32, std::vector<std::pair<int32, Vector<double> > > > &ht) {
  KALDI_ASSERT((direct_diff.Dim() == indirect_diff.Dim()));
  KALDI_ASSERT((direct_diff.Dim() == gmm_.Dim()));

  Vector<double> diff(direct_diff);
  diff.AddVec(1.0, indirect_diff);

  int32 dim = gmm_.Dim();
  std::map<int32, std::vector<std::pair<int32, Vector<double> > > >::const_iterator ht_Iter;
  Matrix<double> tmp(dim, dim + 1);
  tmp.SetZero();

  /// accumulate the p and n statistics
  for (ht_Iter = ht.begin(); ht_Iter != ht.end(); ht_Iter++) {
	int32 cont_index = ht_Iter->first;
	for (int32 i = 0; i < ht_Iter->second.size(); i++) {
      int32 gauss_index = ht_Iter->second[i].first;
	  tmp.AddVecVec(1.0, diff, ht_Iter->second[i].second);

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
  Init(config_.gmm_num_comps, config_.nlength_context_expansion, dim_);
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
  tmp_uint32 = static_cast<uint32>(config_.nlength_context_expansion);
  WriteBasicType(out_stream, binary, tmp_uint32);
  if (!binary) out_stream << "\n";

  if (M_.size() != 0) {
    WriteMarker(out_stream, binary, "<PROJ_MAT>");
    for (int32 i = 0; i < config_.gmm_num_comps; ++i) {
      for (int32 j = 0; j < config_.nlength_context_expansion; ++j) {
        M_[i][j].Write(out_stream, binary);
	  }
    }
  }

  WriteMarker(out_stream, binary, "</FMPE>");
}

void FmpeUpdater::Read(std::istream &in_stream, bool binary,
                         bool add) {
  uint32 tmp_uint32;
  std::string token;

  ExpectMarker(in_stream, binary, "<FMPE>");

  ExpectMarker(in_stream, binary, "<NumGaussians>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 num_gaussians = static_cast<int32>(tmp_uint32);
  ExpectMarker(in_stream, binary, "<LengthContExp>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  int32 length_cont_exp = static_cast<int32>(tmp_uint32);
  
  ReadMarker(in_stream, binary, &token);

  while (token != "</FMPE>") {
    if (token == "<PROJ_MAT>") {
      M_.resize(num_gaussians);
      for (size_t i = 0; i < M_.size(); ++i) {
        M_[i].resize(length_cont_exp);
		for (size_t j = 0; j < M_[i].size(); ++j) {
          M_[i][j].Read(in_stream, binary, add);
		}
      }
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void FmpeUpdater::ComputeAvgStandartDeviation(const AmDiagGmm &am) {
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
			   && M_[0].size() == config_.nlength_context_expansion);
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
