// gmm/fmpe-am-diag-gmm.h

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


#ifndef KALDI_GMM_FMPE_AM_DIAG_GMM_H_
#define KALDI_GMM_FMPE_AM_DIAG_GMM_H_ 1

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mmie-diag-gmm.h"
#include "gmm/ebw-diag-gmm.h"

namespace kaldi {

struct FmpeConfig {
  /// Number of the Gaussian components in the gmm model
  int32 gmm_num_comps;
  /// Number of the Gaussian cluster centers for fast evaluation
  int32 gmm_num_cluster_centers;
  /// the cluster var floor
  BaseFloat cluster_varfloor;
  /// Number of highest-scoring of the best cluster centers
  int32 gmm_cluster_centers_nbest;
  /// Number of highest-scoring of the best gaussians
  int32 gmm_gaussian_nbest;
  /// The lat prob scale
  double lat_prob_scale;
  /// The constant that contrals the overall learning rate
  double E;
  /// The Matrix indicates the length of context expansion
  /// and the weight of each corresponding context frame. e.g.[9][17]
  Matrix<BaseFloat> context_windows;

  /*
    Matrix<BaseFloat> context_windows;
    // Normal dimension is [9][17]
    // Example would be
    // context_windows = [ 0 0 0 0 0 0 0 0 1.0 0 0 0 0 0 0 0 0
    //                     0 0 0 0 0 0 0 0 0 1.0 0 0 0 0 0 0 0
    //  .... etc.
    // Then your nlength_context_expansion variable equals
    // the NumRows() of this.
    // Then you don't have to hard-code the computation in ComputeContExpOffsetFeature.
    // Note: the code in ComputeContExpOffsetFeature that iterates over
    // context_windows will check for zeros, so it will not have to do any work if
    // it finds a zero feature.
    // Also be careful when the same Gaussian index is present on more than one frame,
    // that you are adding the values together, not replacing one with the other or
    // creating duplicates with the same index. [maybe use function DeDuplicateVector(
    //  std::vector<std::pair<int32, Vector<BaseFloat> >*), that would first sort on the
    // int32 and then add together and combine any sets of elements with the same
    // integer value.
  */
  FmpeConfig() {
    gmm_num_comps = 2048;
    gmm_num_cluster_centers = 128;
    cluster_varfloor = 0.01;
    gmm_cluster_centers_nbest = 25;
    gmm_gaussian_nbest = 2;
    lat_prob_scale = 0.083;
    E = 10.0;
  }

  void Register(ParseOptions *po) {
    po->Register("gmm-num-comps", &gmm_num_comps, "Number of the Gaussian"
        " components in the gmm model to calculate the gaussian posteriors.");
    po->Register("gmm-num-cluster-centers", &gmm_num_cluster_centers, "Number"
        " of the Gaussian cluster centers for fast posteriors evaluation.");
    po->Register("cluster-varfloor", &cluster_varfloor,
      "Variance floor used in bottom-up state clustering.");
    po->Register("gmm-cluster-centers-nbest", &gmm_cluster_centers_nbest,
        "Number of highest-scoring of the best cluster centers.");
    po->Register("gmm-gaussian-nbest", &gmm_gaussian_nbest, "Number of"
        " of highest-scoring of the best gaussians.");
    po->Register("lat-prob-scale", &lat_prob_scale,
        "The lattice probability scale, very important.");
    po->Register("E", &E, "The constant that contrals the overall learning rate.");
  }
};

/** \class FmpeAccumModelDiff
 * Class for computing the basic model parameter differentials from
 *  the mpe statistics produced in the first pass of fmpe training
 */
class FmpeAccumModelDiff {
 public:
  FmpeAccumModelDiff(): dim_(0), num_comp_(0) {}
  explicit FmpeAccumModelDiff(const DiagGmm &gmm) {
    Resize(gmm);
  }

  void Read(std::istream &in_stream, bool binary);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_comp, int32 dim);
  /// Calls ResizeAccumulators based on gmm
  void Resize(const DiagGmm &gmm);

  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

  void SetZero();

  // Accessors
  const Vector<double>& mle_occupancy() const { return mle_occupancy_; }
  const Matrix<double>& mean_diff_accumulator() const { return mean_diff_accumulator_; }
  const Matrix<double>& variance_diff_accumulator() const { return variance_diff_accumulator_; }

  /// Computes the Model parameter differentials using the statistics from
  /// the MPE training, including the numerator and denominator accumulators
  /// and applies I-smoothing to the numerator accs, if needed,
  /// which using mle_acc.
  void ComputeModelParaDiff(const DiagGmm &diag_gmm,
                            const AccumDiagGmm &num_acc,
                            const AccumDiagGmm &den_acc,
                            const AccumDiagGmm &mle_acc);


 private:
  int32 dim_;
  int32 num_comp_;

  /// Accumulators
  Vector<double> mle_occupancy_;
  Matrix<double> mean_diff_accumulator_;
  Matrix<double> variance_diff_accumulator_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(FmpeAccumModelDiff);
};

inline void FmpeAccumModelDiff::Resize(const DiagGmm &gmm) {
  Resize(gmm.NumGauss(), gmm.Dim());
}

/** \class FmpeAccs
 *  Class for accumulate the positive and negative statistics
 *  for computing the feature-level minimum phone error estimate of the
 *  parameters of projection M matrix.
 *  The acoustic model is diagonal Gaussian mixture models
 */
class FmpeAccs {
 public:
  explicit FmpeAccs(const FmpeConfig &config)
      : config_(config) {};

  ~FmpeAccs() {}

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Read the am model's parameters differentials
  void ReadModelDiffs(std::istream &in_stream, bool binary);

  /// Initializes the P and N statistics, and model parameter differentials if needed
  void Init(const AmDiagGmm &am_model, bool update);

  /// Initializes the P and N statistics, and diff statistics
  void InitPNandDiff(int32 num_gmm_gauss, int32 con_exp, int32 dim);

  /// Initializes the model parameter differentials
  void InitModelDiff(const AmDiagGmm &model);

  /// Initializes the GMMs for computing the high dimensional features
  void InitializeGMMs(const DiagGmm &gmm, const DiagGmm &gmm_cluster_centers,
                      std::vector<int32> &gaussian_cluster_center_map);

  /// Compute the offset feature given one frame data
  void ComputeOneFrameOffsetFeature(const VectorBase<BaseFloat>& data,
                           std::vector<std::pair<int32, Vector<double> > > *offset) const;

  /// Compute all the offset features given the whole file data
  void ComputeWholeFileOffsetFeature(const MatrixBase<BaseFloat>& data,
                           std::vector<std::vector<std::pair<int32, Vector<double> > > > *whole_file_offset) const;

  /// Compute the context expansion high dimension feature
  /// The high dimension offset feature with the context expansion: "ht";
  /// the vector "ht" store the expanded offset feature corresponding
  /// each context. And each element of "ht" is the relative context's
  /// offset feature, which stored as the pair, including the used
  /// gaussian index and the corresponding offset feature
  /// vector. This structure is designed for the sparse vector ht.
  /// dim is [nContExp * nGaussian * (fea_dim + 1)]
  /// "offset_win" stores the current corresponding offset features
  /// which are used to compute "ht"
  void ComputeContExpOffsetFeature(
       const std::vector<std::vector<std::pair<int32, Vector<double> > >* > &offset_win,
       std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > *ht) const;

  /// obtain the current needed context expension high dimension feature using
  /// the whole file offset features as the inputs which is indexed
  /// by the current frame's number frame_index
  void ComputeHighDimemsionFeature(
       const std::vector<std::vector<std::pair<int32, Vector<double> > > > &whole_file_offset_feat,
       int32 frame_index,
       std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > *ht) const;

  /// Prject the high dimension features down to the dimension of the original
  /// features and add them to the origianl features.
  /// This is the sparse multiply using the non-sparse matrix M and
  /// the sparse high dimension vector ht
  void ProjectHighDimensionFeature(
         const std::vector< std::vector< Matrix<double> > > &M,
         const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht,
         Vector<double> *fea_out) const;

  /// Add the projected feature to the old feature and obtain the new fmpe feature
  void ObtainNewFmpeFeature(const VectorBase<BaseFloat> &data,
         const std::vector< std::vector< Matrix<double> > > &M,
         const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht,
         Vector<double> *fea_new) const;

  /// Accumulate the direct differentials
  void AccumulateDirectDiffFromPosteriors(const DiagGmm &gmm,
                                    const VectorBase<BaseFloat> &data,
                                    const VectorBase<BaseFloat> &posteriors,
                                    Vector<double> *direct_diff);

  /// Accumulate the indirect differentials from posteriors
  void AccumulateInDirectDiffFromPosteriors(const DiagGmm &gmm,
                                      const FmpeAccumModelDiff &fmpe_diaggmm_diff_acc,
                                      const VectorBase<BaseFloat> &data,
                                      const VectorBase<BaseFloat> &posteriors,
                                      Vector<double> *indirect_diff);

  /// Accumulate the indirect differentials from a DiagGmm model
  void AccumulateInDirectDiffFromDiag(const DiagGmm &gmm,
                                      const FmpeAccumModelDiff &fmpe_diaggmm_diff_acc,
                                      const VectorBase<BaseFloat> &data,
                                      BaseFloat frame_posterior,
                                      Vector<double> *indirect_diff);

  /// Accumulate the statistics about the positive and negative differential
  void AccumulateFromDifferential(const VectorBase<double> &direct_diff,
                                  const VectorBase<double> &indirect_diff,
         const std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > &ht);

  // Accessors
  FmpeAccumModelDiff& GetAccsModelDiff(int32 pdf_index);
  const FmpeAccumModelDiff& GetAccsModelDiff(int32 pdf_index) const;

  const std::vector< std::vector< Matrix<double> > >& pos() const { return p_; }
  const std::vector< std::vector< Matrix<double> > >& neg() const { return n_; }
  const FmpeConfig& config() const { return config_; }

  /// Returns the number of mixture components in the GMM model
  int32 NumGaussInGmm() const { return gmm_.NumGauss(); }
  /// Returns the number of cluster centers in the cluster center GMM
  int32 NumClusterCenter() const { return gmm_cluster_centers_.NumGauss(); }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

 private:
  FmpeConfig config_;
  /// These contain the gmm models used to calculate the high deminsion
  /// offet feature : one compute the high dimension vector gaussian posteriors,
  /// and the other one is just for more efficient computing using
  /// the most likely cluster centers
  DiagGmm gmm_;
  DiagGmm gmm_cluster_centers_;

  /// The mapping between the gmm_ model and the cluster centers of gmm_cluster_centers_
  std::vector<int32> gaussian_cluster_center_map_;

  /// The basic model parameter differentials for the AmDiagGmm
  std::vector<FmpeAccumModelDiff*> model_diff_accumulators_;

  /// The positive accumulated matrix p_ij; dim is [nGauss][nContExp][fea_dim][fea_dim + 1].
  std::vector< std::vector< Matrix<double> > > p_;
  /// The negative accumulated matrix n_ij; dim is [nGauss][nContExp][fea_dim][fea_dim + 1].
  std::vector< std::vector< Matrix<double> > > n_;
  /// The summation of the differential
  Vector<double> diff_;
  /// The summation of the direct differential
  Vector<double> direct_diff_;
  /// The summation of the indirect differential
  Vector<double> indirect_diff_;

  /// The feature dim
  int32 dim_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(FmpeAccs);
};

inline FmpeAccumModelDiff& FmpeAccs::GetAccsModelDiff(int32 pdf_index) {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < model_diff_accumulators_.size())
               && (model_diff_accumulators_[pdf_index] != NULL));
  return *(model_diff_accumulators_[pdf_index]);
}

inline const FmpeAccumModelDiff& FmpeAccs::GetAccsModelDiff(int32 pdf_index) const {
  KALDI_ASSERT((static_cast<size_t>(pdf_index) < model_diff_accumulators_.size())
               && (model_diff_accumulators_[pdf_index] != NULL));
  return *(model_diff_accumulators_[pdf_index]);
}

/** \class FmpeUpdater
 *  Class for containing the functions that updating the feature-level
 *  minimum phone error estimate of the parameters of projection M matrix
 *  that adds offsets to the original feature.
 *  The acoustic model is diagonal Gaussian mixture models
 */
class FmpeUpdater {
 public:
  explicit FmpeUpdater(const FmpeAccs &accs);
  ~FmpeUpdater() {}

  // provide copy constructor.
  explicit FmpeUpdater(const FmpeUpdater &other);

  void Read(std::istream &in_stream, bool binary);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Initializes feature projection Matrix M
  void Init(int32 num_gmm_gauss, int32 con_exp, int32 dim);

  /// compute the average standard deviation of gaussians
  /// in the current AmDiagGmm set
  void ComputeAvgStandardDeviation(const AmDiagGmm &am);

  /// Update the projection matrix M
  void Update(const FmpeAccs &accs,
              BaseFloat *obj_change_out,
              BaseFloat *count_out);

  // Accessors
  const std::vector< std::vector< Matrix<double> > >& ProjMat() const { return M_; }
  const FmpeConfig& config() const { return config_; }

 private:
  FmpeConfig config_;

  /// The average standard deviation of gaussians in the current AmDiagGmm set
  Vector<double> avg_std_var_;

  /// The feature projection matrix; dim is [nGauss][nContExp][fea_dim][fea_dim + 1].
  std::vector< std::vector< Matrix<double> > > M_;

  /// The feature dim
  int32 dim_;
};

/** Clusters the Gaussians in the gmm model to some cluster centers
 */
void ClusterGmmToClusterCenters(const DiagGmm &gmm,
                                int32 num_cluster_centers,
                                BaseFloat cluster_varfloor,
                                DiagGmm *ubm_cluster_centers,
                                std::vector<int32> *cluster_center_map);

/** First clusters the Gaussians in an acoustic model to a single GMM with specified
 * number of components. Using the same algorithm in the SGMM's UBM
 * initialization, and then Clusters the Gaussians in the gmm model
 * to some cluster centers, which is for more efficient evaluation of the
 * gaussian posteriors just with the most likely cluster centers
 */
void ObtainUbmAndSomeClusterCenters(
                     const AmDiagGmm &am,
                     const Vector<BaseFloat> &state_occs,
                     const FmpeConfig &config,
                     DiagGmm *gmm_out,
                     DiagGmm *gmm_cluster_centers_out,
                     std::vector<int32> *gaussian_cluster_center_map_out);


}  // End namespace kaldi


#endif  // KALDI_GMM_FMPE_AM_DIAG_GMM_H_
