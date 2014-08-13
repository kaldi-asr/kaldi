// ivector/plda.h

// Copyright 2013    Daniel Povey


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

#ifndef KALDI_IVECTOR_PLDA_H_
#define KALDI_IVECTOR_PLDA_H_

#include <vector>
#include <algorithm>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"

namespace kaldi {

/* This code
   implements Probabilistic Linear Discriminant Analysis: see
    "Probabilistic Linear Discriminant Analysis" by Sergey Ioffe, ECCV 2006.
   At least, that was the inspiration.  The E-M is an efficient method
   that I derived myself (note: it could be made even more efficient but
   it doesn't seem to be necessary as it's already very fast).

   This implementation of PLDA only supports estimating with a between-class
   dimension equal to the feature dimension.  If you want a between-class
   covariance that has a lower dimension, you can just remove the smallest
   elements of the diagonalized between-class covariance matrix.  This is not
   100% exact (wouldn't give you as good likelihood as E-M estimation with that
   dimension) but it's close enough.  */

struct PldaConfig {
  // This config is for the application of PLDA as a transform to iVectors,
  // prior to dot-product scoring.
  bool normalize_length;
  PldaConfig(): normalize_length(true) { }
  void Register(OptionsItf *po) {
    po->Register("normalize-length", &normalize_length,
                 "If true, do length normalization as part of PLDA (see code for "
                 "details)");
  }
};


class Plda {
 public:
  Plda() { }


  /// Transforms iVector into the space where the within-class variance
  /// is unit and between-class variance is diagonalized.  The only
  /// anticipated use of this function is to pre-transform iVectors
  /// before giving them to the function LogLikelihoodRatio (it's
  /// done this way for efficiency because a given iVector may be
  /// used multiple times in LogLikelihoodRatio and we don't want
  /// do repeat the matrix multiplication
  /// 
  /// If config.normalize_length == true, it will also normalize the iVector's
  /// length by multiplying by a scalar that ensures that ivector^T inv_var
  /// ivector = dim.  In this case, "num_examples" comes into play because it
  /// affects the expected covariance matrix of the iVector.  The normalization
  /// factor is returned, even if config.normalize_length == false, in which
  /// case the normalization factor is computed but not applied.
  double TransformIvector(const PldaConfig &config,
                          const VectorBase<double> &ivector,
                          int32 num_examples,
                          VectorBase<double> *transformed_ivector) const;

  /// float version of the above (not BaseFloat because we'd be implementing it
  /// twice for the same type if BaseFloat == double).
  float TransformIvector(const PldaConfig &config,
                         const VectorBase<float> &ivector,
                         int32 num_examples,
                         VectorBase<float> *transformed_ivector) const;
  
  /// Returns the log-likelihood ratio
  /// log (p(test_ivector | same) / p(test_ivector | different)).
  /// transformed_train_ivector is an average over utterances for
  /// that speaker.  Both transformed_train_vector and transformed_test_ivector
  /// are assumed to have been transformed by the function TransformIvector().
  /// Note: any length normalization will have been done while computing
  /// the transformed iVectors.
  double LogLikelihoodRatio(const VectorBase<double> &transformed_train_ivector,
                            int32 num_train_utts,
                            const VectorBase<double> &transformed_test_ivector);

  
  /// This function smooths the within-class covariance by adding to it,
  /// smoothing_factor (e.g. 0.1) times the between-class covariance (it's
  /// implemented by modifying transform_).  This is to compensate for
  /// situations where there were too few utterances per speaker get a good
  /// estimate of the within-class covariance, and where the leading elements of
  /// psi_ were as a result very large.
  void SmoothWithinClassCovariance(double smoothing_factor);
  
  int32 Dim() const { return mean_.Dim(); }
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
 protected:
  void ComputeDerivedVars(); // computes offset_.
  friend class PldaEstimator;
  friend class PldaUnsupervisedAdaptor;
  
  Vector<double> mean_;  // mean of samples in original space.
  Matrix<double> transform_; // of dimension Dim() by Dim();
                             // this transform makes within-class covar unit
                             // and diagonalizes the between-class covar.
  Vector<double> psi_; // of dimension Dim().  The between-class
                       // (diagonal) covariance elements, in decreasing order.

  Vector<double> offset_;  // derived variable: -1.0 * transform_ * mean_

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Plda);

  /// This returns a normalization factor, which is a quantity we
  /// must multiply "transformed_ivector" by so that it has the length
  /// that it "should" have.  We assume "transformed_ivector" is an
  /// iVector in the transformed space (i.e., mean-subtracted, and
  /// multiplied by transform_).  The covariance it "should" have
  /// in this space is \Psi + I/num_examples.
  double GetNormalizationFactor(const VectorBase<double> &transformed_ivector,
                                int32 num_examples) const;
  
};


class PldaStats {
 public:
  PldaStats(): dim_(0) { } /// The dimension is set up the first time you add samples.

  /// This function adds training samples corresponding to
  /// one class (e.g. a speaker).  Each row is a separate
  /// sample from this group.  The "weight" would normally
  /// be 1.0, but you can set it to other values if you want
  /// to weight your training samples.
  void AddSamples(double weight,
                  const Matrix<double> &group);
    
  int32 Dim() const { return dim_; }

  void Init(int32 dim);

  void Sort() { std::sort(class_info_.begin(), class_info_.end()); }
  bool IsSorted() const;
  ~PldaStats();
 protected:
  
  friend class PldaEstimator;
  
  int32 dim_;
  int64 num_classes_;
  int64 num_examples_; // total number of examples, sumed over classes.
  double class_weight_; // total over classes, of their weight.
  double example_weight_; // total over classes, of weight times #examples.

  Vector<double> sum_; // Weighted sum of class means (normalize by class_weight_
                       // to get mean).

  SpMatrix<double> offset_scatter_; // Sum over all examples, of the weight
                                    // times (example - class-mean).
  
  // We have one of these objects per class.
  struct ClassInfo {
    double weight;
    Vector<double> *mean; // owned here, but as a pointer so
                          // sort can be lightweight
    int32 num_examples; // the number of examples in the class
    bool operator < (const ClassInfo &other) const {
      return (num_examples < other.num_examples);
    }
    ClassInfo(double weight, Vector<double> *mean, int32 num_examples):
        weight(weight), mean(mean), num_examples(num_examples) { }
  };
   
  std::vector<ClassInfo> class_info_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PldaStats);
};


struct PldaEstimationConfig {
  int32 num_em_iters;
  PldaEstimationConfig(): num_em_iters(10){ }
  void Register(OptionsItf *po) {
    po->Register("num-em-iters", &num_em_iters,
                 "Number of iterations of E-M used for PLDA estimation");
  }
};

class PldaEstimator {
 public:
  PldaEstimator(const PldaStats &stats);
  
  void Estimate(const PldaEstimationConfig &config,
                Plda *output);
private:
  typedef PldaStats::ClassInfo ClassInfo;
  
  /// Returns the part of the objf relating to
  /// offsets from the class means.  (total, not normalized)
  double ComputeObjfPart1() const;
  
  /// Returns the part of the obj relating to
  /// the class means (total_not normalized)
  double ComputeObjfPart2() const;

  /// Returns the objective-function per sample.
  double ComputeObjf() const;

  int32 Dim() const { return stats_.Dim(); }

  void EstimateOneIter();
  
  void InitParameters();

  void ResetPerIterStats();

  // gets stats from intra-class variation (stats_.offset_scatter_).
  void GetStatsFromIntraClass();

  // gets part of stats relating to class means.
  void GetStatsFromClassMeans();

  // M-step
  void EstimateFromStats();

  // Copy to output.
  void GetOutput(Plda *plda);
  
  const PldaStats &stats_;

  SpMatrix<double> within_var_;
  SpMatrix<double> between_var_;

  // These stats are reset on each iteration.
  SpMatrix<double> within_var_stats_;
  double within_var_count_; // count corresponding to within_var_stats_
  SpMatrix<double> between_var_stats_;
  double between_var_count_; // count corresponding to within_var_stats_

  KALDI_DISALLOW_COPY_AND_ASSIGN(PldaEstimator);
};



struct PldaUnsupervisedAdaptorConfig {
  BaseFloat mean_diff_scale;
  BaseFloat within_covar_scale;
  BaseFloat between_covar_scale;
  
  PldaUnsupervisedAdaptorConfig():
      mean_diff_scale(1.0),
      within_covar_scale(0.3),
      between_covar_scale(0.7) { }

  void Register(OptionsItf *po) {
    po->Register("mean-diff-scale", &mean_diff_scale,
                 "Scale with which to add to the total data variance, the outer "
                 "product of the difference between the original mean and the "
                 "adaptation-data mean");
    po->Register("within-covar-scale", &within_covar_scale,
                 "Scale that determines how much of excess variance in a "
                 "particular direction gets attributed to within-class covar.");
    po->Register("between-covar-scale", &between_covar_scale,
                 "Scale that determines how much of excess variance in a "
                 "particular direction gets attributed to between-class covar.");

  }
};

/**
  This class takes unlabeled iVectors from the domain of interest and uses their
  mean and variance to adapt your PLDA matrices to a new domain.  This class
  also stores stats for this form of adaptation.  */
class PldaUnsupervisedAdaptor {
 public:
  PldaUnsupervisedAdaptor(): tot_weight_(0.0) { }
  // Add stats to this class.  Normally the weight will be 1.0.
  void AddStats(double weight, const Vector<double> &ivector);
  void AddStats(double weight, const Vector<BaseFloat> &ivector);
  

  void UpdatePlda(const PldaUnsupervisedAdaptorConfig &config,
                  Plda *plda) const;
 private:

  double tot_weight_;
  Vector<double> mean_stats_;
  SpMatrix<double> variance_stats_;    
};



}  // namespace kaldi

#endif
