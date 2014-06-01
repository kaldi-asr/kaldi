// gmm/am-diag-gmm.cc

// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (Author: Daniel Povey)  Karel Vesely
// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Georg Stemmer

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

#include <queue>
#include <string>
#include <vector>
using std::vector;

#include "gmm/am-diag-gmm.h"
#include "util/stl-utils.h"
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"

namespace kaldi {

AmDiagGmm::~AmDiagGmm() {
  DeletePointers(&densities_);
}

void AmDiagGmm::Init(const DiagGmm &proto, int32 num_pdfs) {
  if (densities_.size() != 0) {
    KALDI_WARN << "Init() called on a non-empty object. Contents will be "
        "overwritten";
    DeletePointers(&densities_);
  }
  if (num_pdfs == 0) {
    KALDI_WARN << "Init() called with number of pdfs = 0. Will do nothing.";
    return;
  }

  densities_.resize(num_pdfs, NULL);
  for (vector<DiagGmm*>::iterator itr = densities_.begin(),
      end = densities_.end(); itr != end; ++itr) {
    *itr = new DiagGmm();
    (*itr)->CopyFromDiagGmm(proto);
  }
}

void AmDiagGmm::AddPdf(const DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    KALDI_ASSERT(gmm.Dim() == this->Dim());

  DiagGmm *gmm_ptr = new DiagGmm();
  gmm_ptr->CopyFromDiagGmm(gmm);
  densities_.push_back(gmm_ptr);
}

void AmDiagGmm::RemovePdf(int32 pdf_index) {
  KALDI_ASSERT(static_cast<size_t>(pdf_index) < densities_.size());
  delete densities_[pdf_index];
  densities_.erase(densities_.begin() + pdf_index);
}

int32 AmDiagGmm::NumGauss() const {
  int32 ans = 0;
  for (size_t i = 0; i < densities_.size(); i++)
    ans += densities_[i]->NumGauss();
  return ans;
}

void AmDiagGmm::CopyFromAmDiagGmm(const AmDiagGmm &other) {
  if (densities_.size() != 0) {
    DeletePointers(&densities_);
  }
  densities_.resize(other.NumPdfs(), NULL);
  for (int32 i = 0, end = densities_.size(); i < end; i++) {
    densities_[i] = new DiagGmm();
    densities_[i]->CopyFromDiagGmm(*other.densities_[i]);
  }
}

int32 AmDiagGmm::ComputeGconsts() {
  int32 num_bad = 0;
  for (std::vector<DiagGmm*>::iterator itr = densities_.begin(),
      end = densities_.end(); itr != end; ++itr) {
    num_bad += (*itr)->ComputeGconsts();
  }
  if (num_bad > 0)
    KALDI_WARN << "Found " << num_bad << " Gaussian components.";
  return num_bad;
}


void AmDiagGmm::SplitByCount(const Vector<BaseFloat> &state_occs,
                             int32 target_components,
                             float perturb_factor, BaseFloat power,
                             BaseFloat min_count) {
  int32 gauss_at_start = NumGauss();
  std::vector<int32> targets;
  GetSplitTargets(state_occs, target_components, power,
                  min_count, &targets);

  for (int32 i = 0; i < NumPdfs(); i++) {
    if (densities_[i]->NumGauss() < targets[i])
      densities_[i]->Split(targets[i], perturb_factor);
  }

  KALDI_LOG << "Split " << NumPdfs() << " states with target = "
            << target_components << ", power = " << power
            << ", perturb_factor = " << perturb_factor
            << " and min_count = " << min_count
            << ", split #Gauss from " << gauss_at_start << " to "
            << NumGauss();
}


void AmDiagGmm::MergeByCount(const Vector<BaseFloat> &state_occs,
                             int32 target_components,
                             BaseFloat power,
                             BaseFloat min_count) {
  int32 gauss_at_start = NumGauss();
  std::vector<int32> targets;
  GetSplitTargets(state_occs, target_components,
                  power, min_count, &targets);

  for (int32 i = 0; i < NumPdfs(); i++) {
    if (targets[i] == 0) targets[i] = 1;  // can't merge below 1.
    if (densities_[i]->NumGauss() > targets[i])
      densities_[i]->Merge(targets[i]);
  }

  KALDI_LOG << "Merged " << NumPdfs() << " states with target = "
            << target_components << ", power = " << power
            << " and min_count = " << min_count
            << ", merged from " << gauss_at_start << " to "
            << NumGauss();
}

void AmDiagGmm::Read(std::istream &in_stream, bool binary) {
  int32 num_pdfs, dim;

  ExpectToken(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim);
  ExpectToken(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  densities_.reserve(num_pdfs);
  for (int32 i = 0; i < num_pdfs; i++) {
    densities_.push_back(new DiagGmm());
    densities_.back()->Read(in_stream, binary);
    KALDI_ASSERT(densities_.back()->Dim() == dim);
  }
}

void AmDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  int32 dim = this->Dim();
  if (dim == 0) {
    KALDI_WARN << "Trying to write empty AmDiagGmm object.";
  }
  WriteToken(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim);
  WriteToken(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, static_cast<int32>(densities_.size()));
  for (std::vector<DiagGmm*>::const_iterator it = densities_.begin(),
      end = densities_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

void UbmClusteringOptions::Check() {
  if (ubm_num_gauss > intermediate_num_gauss)
    KALDI_ERR << "Invalid parameters: --ubm-num_gauss=" << ubm_num_gauss
              << " > --intermediate-num_gauss=" << intermediate_num_gauss;
  if (ubm_num_gauss > max_am_gauss)
    KALDI_ERR << "Invalid parameters: --ubm-num_gauss=" << ubm_num_gauss
              << " > --max-am-gauss=" << max_am_gauss;
  if (ubm_num_gauss <= 0)
    KALDI_ERR << "Invalid parameters: --ubm-num_gauss=" << ubm_num_gauss;
  if (cluster_varfloor <= 0)
    KALDI_ERR << "Invalid parameters: --cluster-varfloor="
              << cluster_varfloor;
  if (reduce_state_factor <= 0 || reduce_state_factor > 1)
    KALDI_ERR << "Invalid parameters: --reduce-state-factor="
              << reduce_state_factor;
}

void ClusterGaussiansToUbm(const AmDiagGmm &am,
                           const Vector<BaseFloat> &state_occs,
                           UbmClusteringOptions opts,
                           DiagGmm *ubm_out) {
  opts.Check();  // Make sure the various # of Gaussians make sense.
  if (am.NumGauss() > opts.max_am_gauss) {
    KALDI_LOG << "ClusterGaussiansToUbm: first reducing num-gauss from " << am.NumGauss()
              << " to " << opts.max_am_gauss;
    AmDiagGmm tmp_am;
    tmp_am.CopyFromAmDiagGmm(am);
    BaseFloat power = 1.0, min_count = 1.0; // Make the power 1, which I feel
    // is appropriate to the way we're doing the overall clustering procedure.
    tmp_am.MergeByCount(state_occs, opts.max_am_gauss, power, min_count);

    if (tmp_am.NumGauss() > opts.max_am_gauss) {
      KALDI_LOG << "Clustered down to " << tmp_am.NumGauss()
                << "; will not cluster further";
      opts.max_am_gauss = tmp_am.NumGauss();
    }
    ClusterGaussiansToUbm(tmp_am, state_occs, opts, ubm_out);
    return;
  }
  
  int32 num_pdfs = static_cast<int32>(am.NumPdfs()),
      dim = am.Dim(),
      num_clust_states = static_cast<int32>(opts.reduce_state_factor*num_pdfs);

  Vector<BaseFloat> tmp_mean(dim);
  Vector<BaseFloat> tmp_var(dim);
  DiagGmm tmp_gmm;
  vector<Clusterable*> states;
  states.reserve(num_pdfs);  // NOT resize(); uses push_back.

  // Replace the GMM for each state with a single Gaussian.
  KALDI_VLOG(1) << "Merging densities to 1 Gaussian per state.";
  for (int32 pdf_index = 0; pdf_index < num_pdfs; pdf_index++) {
    KALDI_VLOG(3) << "Merging Gausians for state : " << pdf_index;
    tmp_gmm.CopyFromDiagGmm(am.GetPdf(pdf_index));
    tmp_gmm.Merge(1);
    tmp_gmm.GetComponentMean(0, &tmp_mean);
    tmp_gmm.GetComponentVariance(0, &tmp_var);
    tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
    BaseFloat this_weight = state_occs(pdf_index);
    tmp_mean.Scale(this_weight);
    tmp_var.Scale(this_weight);
    states.push_back(new GaussClusterable(tmp_mean, tmp_var,
                          opts.cluster_varfloor, this_weight));
  }

  // Bottom-up clustering of the Gaussians corresponding to each state, which
  // gives a partial clustering of states in the 'state_clusters' vector.
  vector<int32> state_clusters;
  KALDI_VLOG(1) << "Creating " << num_clust_states << " clusters of states.";
  ClusterBottomUp(states, std::numeric_limits<BaseFloat>::max(), num_clust_states,
                  NULL /*actual clusters not needed*/,
                  &state_clusters /*get the cluster assignments*/);
  DeletePointers(&states);

  // For each cluster of states, create a pool of all the Gaussians in those
  // states, weighted by the state occupancies. This is done so that initially
  // only the Gaussians corresponding to "similar" states (similarity as
  // determined by the previous clustering) are merged.
  vector< vector<Clusterable*> > state_clust_gauss;
  state_clust_gauss.resize(num_clust_states);
  for (int32 pdf_index = 0; pdf_index < num_pdfs; pdf_index++) {
    int32 current_cluster = state_clusters[pdf_index];
    for (int32 num_gauss = am.GetPdf(pdf_index).NumGauss(),
        gauss_index = 0; gauss_index < num_gauss; ++gauss_index) {
      am.GetGaussianMean(pdf_index, gauss_index, &tmp_mean);
      am.GetGaussianVariance(pdf_index, gauss_index, &tmp_var);
      tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
      BaseFloat this_weight =  state_occs(pdf_index) *
          (am.GetPdf(pdf_index).weights())(gauss_index);
      tmp_mean.Scale(this_weight);
      tmp_var.Scale(this_weight);
      state_clust_gauss[current_cluster].push_back(new GaussClusterable(
          tmp_mean, tmp_var, opts.cluster_varfloor, this_weight));
    }
  }

  // This is an unlikely operating scenario, no need to handle this in a more
  // optimized fashion.
  if (opts.intermediate_num_gauss > am.NumGauss()) {
    KALDI_WARN << "Intermediate num_gauss " << opts.intermediate_num_gauss
               << " is more than num-gauss " << am.NumGauss()
               << ", reducing it to " << am.NumGauss();
    opts.intermediate_num_gauss = am.NumGauss();
  }

  // The compartmentalized clusterer used below does not merge compartments.
  if (opts.intermediate_num_gauss < num_clust_states) {
    KALDI_WARN << "Intermediate num_gauss " << opts.intermediate_num_gauss
               << " is less than # of preclustered states " << num_clust_states
               << ", increasing it to " << num_clust_states;
    opts.intermediate_num_gauss = num_clust_states;
  }
    
  KALDI_VLOG(1) << "Merging from " << am.NumGauss() << " Gaussians in the "
                << "acoustic model, down to " << opts.intermediate_num_gauss
                << " Gaussians.";
  vector< vector<Clusterable*> > gauss_clusters_out;
  ClusterBottomUpCompartmentalized(state_clust_gauss, std::numeric_limits<BaseFloat>::max(),
                                   opts.intermediate_num_gauss,
                                   &gauss_clusters_out, NULL);
  for (int32 clust_index = 0; clust_index < num_clust_states; clust_index++)
    DeletePointers(&state_clust_gauss[clust_index]);

  // Next, put the remaining clustered Gaussians into a single GMM.
  KALDI_VLOG(1) << "Putting " << opts.intermediate_num_gauss << " Gaussians "
                << "into a single GMM for final merge step.";
  Matrix<BaseFloat> tmp_means(opts.intermediate_num_gauss, dim);
  Matrix<BaseFloat> tmp_vars(opts.intermediate_num_gauss, dim);
  Vector<BaseFloat> tmp_weights(opts.intermediate_num_gauss);
  Vector<BaseFloat> tmp_vec(dim);
  int32 gauss_index = 0;
  for (int32 clust_index = 0; clust_index < num_clust_states; clust_index++) {
    for (int32 i = gauss_clusters_out[clust_index].size()-1; i >=0; --i) {
      GaussClusterable *this_cluster = static_cast<GaussClusterable*>(
          gauss_clusters_out[clust_index][i]);
      BaseFloat weight = this_cluster->count();
      KALDI_ASSERT(weight > 0);
      tmp_weights(gauss_index) = weight;
      tmp_vec.CopyFromVec(this_cluster->x_stats());
      tmp_vec.Scale(1/weight);
      tmp_means.CopyRowFromVec(tmp_vec, gauss_index);
      tmp_vec.CopyFromVec(this_cluster->x2_stats());
      tmp_vec.Scale(1/weight);
      tmp_vec.AddVec2(-1.0, tmp_means.Row(gauss_index));  // x^2 stats to var.
      tmp_vars.CopyRowFromVec(tmp_vec, gauss_index);
      gauss_index++;
    }
    DeletePointers(&(gauss_clusters_out[clust_index]));
  }
  tmp_gmm.Resize(opts.intermediate_num_gauss, dim);
  tmp_weights.Scale(1.0/tmp_weights.Sum());
  tmp_gmm.SetWeights(tmp_weights);
  tmp_vars.InvertElements();  // need inverse vars...
  tmp_gmm.SetInvVarsAndMeans(tmp_vars, tmp_means);

  // Finally, cluster to the desired number of Gaussians in the UBM.
  if (opts.ubm_num_gauss < tmp_gmm.NumGauss()) {
    tmp_gmm.Merge(opts.ubm_num_gauss);
    KALDI_VLOG(1) << "Merged down to " << tmp_gmm.NumGauss() << " Gaussians.";
  } else {
    KALDI_WARN << "Not merging Gaussians since " << opts.ubm_num_gauss
               << " < " << tmp_gmm.NumGauss();
  }
  ubm_out->CopyFromDiagGmm(tmp_gmm);
}

}  // namespace kaldi
