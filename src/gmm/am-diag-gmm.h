// gmm/am-diag-gmm.h

// Copyright 2009-2012  Saarland University (Author:  Arnab Ghoshal)
//                      Johns Hopkins University (Author: Daniel Povey)
//                      Karel Vesely

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

#ifndef KALDI_GMM_AM_DIAG_GMM_H_
#define KALDI_GMM_AM_DIAG_GMM_H_ 1

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/diag-gmm.h"
#include "itf/options-itf.h"

namespace kaldi {
/// @defgroup DiagGmm DiagGmm
/// @{
/// kaldi Diagonal Gaussian Mixture Models

class AmDiagGmm {
 public:
  AmDiagGmm() {}
  ~AmDiagGmm();

  /// Initializes with a single "prototype" GMM.
  void Init(const DiagGmm &proto, int32 num_pdfs);
  /// Adds a GMM to the model, and increments the total number of PDFs.
  void AddPdf(const DiagGmm &gmm);
  /// Copies the parameters from another model. Allocates necessary memory.
  void CopyFromAmDiagGmm(const AmDiagGmm &other);

  void SplitPdf(int32 idx, int32 target_components, float perturb_factor);

  // In SplitByCount we use the "target_components" and "power"
  // to work out targets for each state (according to power-of-occupancy rule),
  // and any state less than its target gets mixed up.  If some states
  // were over their target, this may take the #Gauss over the target.
  // we enforce a min-count on Gaussians while splitting (don't split
  // if it would take it below min-count).
  void SplitByCount(const Vector<BaseFloat> &state_occs,
                    int32 target_components, float perturb_factor,
                    BaseFloat power, BaseFloat min_count);


  // In MergeByCount we use the "target_components" and "power"
  // to work out targets for each state (according to power-of-occupancy rule),
  // and any state over its target gets mixed down.  If some states
  // were under their target, this may take the #Gauss below the target.
  void MergeByCount(const Vector<BaseFloat> &state_occs,
                    int32 target_components,
                    BaseFloat power, BaseFloat min_count);

  /// Sets the gconsts for all the PDFs. Returns the total number of Gaussians
  /// over all PDFs that are "invalid" e.g. due to zero weights or variances.
  int32 ComputeGconsts();

  BaseFloat LogLikelihood(const int32 pdf_index,
                          const VectorBase<BaseFloat> &data) const;
  
  void Read(std::istream &in_stream, bool binary);
  void Write(std::ostream &out_stream, bool binary) const;

  int32 Dim() const {
    return (densities_.size() > 0)? densities_[0]->Dim() : 0;
  }
  int32 NumPdfs() const { return densities_.size(); }
  int32 NumGauss() const;
  int32 NumGaussInPdf(int32 pdf_index) const;

  /// Accessors
  DiagGmm& GetPdf(int32 pdf_index);
  const DiagGmm& GetPdf(int32 pdf_index) const;
  void GetGaussianMean(int32 pdf_index, int32 gauss,
                       VectorBase<BaseFloat> *out) const;
  void GetGaussianVariance(int32 pdf_index, int32 gauss,
                           VectorBase<BaseFloat> *out) const;

  /// Mutators
  void SetGaussianMean(int32 pdf_index, int32 gauss_index,
                       const VectorBase<BaseFloat> &in);

 private:
  std::vector<DiagGmm*> densities_;
//  int32 dim_;

  void RemovePdf(int32 pdf_index);

  KALDI_DISALLOW_COPY_AND_ASSIGN(AmDiagGmm);
};


inline BaseFloat AmDiagGmm::LogLikelihood(
    const int32 pdf_index, const VectorBase<BaseFloat> &data) const {
  return densities_[pdf_index]->LogLikelihood(data);
}

inline int32 AmDiagGmm::NumGaussInPdf(int32 pdf_index) const {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
      && (densities_[pdf_index] != NULL));
  return densities_[pdf_index]->NumGauss();
}

inline DiagGmm& AmDiagGmm::GetPdf(int32 pdf_index) {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
               && (densities_[pdf_index] != NULL));
  return *(densities_[pdf_index]);
}

inline const DiagGmm& AmDiagGmm::GetPdf(int32 pdf_index) const {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
               && (densities_[pdf_index] != NULL));
  return *(densities_[pdf_index]);
}

inline void AmDiagGmm::GetGaussianMean(int32 pdf_index, int32 gauss,
                                       VectorBase<BaseFloat> *out) const {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
      && (densities_[pdf_index] != NULL));
  densities_[pdf_index]->GetComponentMean(gauss, out);
}

inline void AmDiagGmm::GetGaussianVariance(int32 pdf_index, int32 gauss,
                                           VectorBase<BaseFloat> *out) const {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
               && (densities_[pdf_index] != NULL));
  densities_[pdf_index]->GetComponentVariance(gauss, out);
}

inline void AmDiagGmm::SetGaussianMean(int32 pdf_index, int32 gauss_index,
                                       const VectorBase<BaseFloat> &in) {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
               && (densities_[pdf_index] != NULL));
  densities_[pdf_index]->SetComponentMean(gauss_index, in);
}

inline void AmDiagGmm::SplitPdf(int32 pdf_index,
                                           int32 target_components,
                                           float perturb_factor) {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < densities_.size())
               && (densities_[pdf_index] != NULL));
  densities_[pdf_index]->Split(target_components, perturb_factor);
}

struct UbmClusteringOptions {
  int32 ubm_num_gauss;
  BaseFloat reduce_state_factor;
  int32 intermediate_num_gauss;
  BaseFloat cluster_varfloor;
  int32 max_am_gauss;

  UbmClusteringOptions()
      : ubm_num_gauss(400), reduce_state_factor(0.2),
        intermediate_num_gauss(4000), cluster_varfloor(0.01),
        max_am_gauss(20000) {}
  UbmClusteringOptions(int32 ncomp, BaseFloat red, int32 interm_gauss,
                       BaseFloat vfloor, int32 max_am_gauss)
        : ubm_num_gauss(ncomp), reduce_state_factor(red),
          intermediate_num_gauss(interm_gauss), cluster_varfloor(vfloor),
          max_am_gauss(max_am_gauss) {}
  void Register(OptionsItf *opts) {
    std::string module = "UbmClusteringOptions: ";
    opts->Register("max-am-gauss", &max_am_gauss, module+
                   "We first reduce acoustic model to this max #Gauss before clustering.");
    opts->Register("ubm-num-gauss", &ubm_num_gauss, module+
                   "Number of Gaussians components in the final UBM.");
    opts->Register("ubm-numcomps", &ubm_num_gauss, module+
                   "Backward compatibility option (see ubm-num-gauss)");
    opts->Register("reduce-state-factor", &reduce_state_factor, module+
                   "Intermediate number of clustered states (as fraction of total states).");
    opts->Register("intermediate-num-gauss", &intermediate_num_gauss, module+
                   "Intermediate number of merged Gaussian components.");
    opts->Register("intermediate-numcomps", &intermediate_num_gauss, module+
                   "Backward compatibility option (see intermediate-num-gauss)");
    opts->Register("cluster-varfloor", &cluster_varfloor, module+
                   "Variance floor used in bottom-up state clustering.");
  }

  void Check();
};

/** Clusters the Gaussians in an acoustic model to a single GMM with specified
 *  number of components. First the each state is mixed-down to a single
 *  Gaussian, then the states are clustered by clustering these Gaussians in a
 *  bottom-up fashion. Number of clusters is determined by reduce_state_factor.
 *  The Gaussians for each cluster of states are then merged based on the least
 *  likelihood reduction till there are intermediate_numcomp Gaussians, which
 *  are then merged into ubm_num_gauss Gaussians.
 *  This is the UBM initialization algorithm described in section 2.1 of Povey,
 *  et al., "The subspace Gaussian mixture model - A structured model for speech
 *  recognition", In Computer Speech and Language, April 2011.
 */
void ClusterGaussiansToUbm(const AmDiagGmm &am,
                           const Vector<BaseFloat> &state_occs,
                           UbmClusteringOptions opts,
                           DiagGmm *ubm_out);




}  // namespace kaldi

/// @} DiagGmm
#endif  // KALDI_GMM_AM_DIAG_GMM_H_
