// adapt/differentiable-fmllr.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_DIFFERENTIABLE_FMLLR_H_
#define KALDI_TRANSFORM_DIFFERENTIABLE_FMLLR_H_

#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "hmm/posterior.h"
#include "matrix/matrix-functions.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"

namespace kaldi {
namespace differentiable_transform {


// This header contains some utilities for implementing differentiable fMLLR.
// Since it is fairly complicated, we aren't putting all the implementation
// details in class FmllrTransform (in differentiable-transform.h), but
// segregating most of the technical stuff to this file.  This also
// allows us to separate out the testing of individual components.
// The reference for things in this header is
// http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf.
// The notation we are using corresponds to the notation used in
// the "Summary" section of that document.




/**
   With reference to the notation in
     http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf,
   this class implements the operation that takes G and K as input (and the
   count gamma), and produces A.  This has been separated into its own object
   for purposes of testability.
 */
struct FmllrEstimatorOptions {

  // singular_value_relative_floor is floor that we apply on the
  // singular values of the inputs G and K, to ensure that no NaN's are
  // generated in the forward pass and to prevent the derivatives
  // in the backprop from becoming undefined.  It affects both
  // the forward and backward computations.  A warning will be printed
  // if this floor actually had an effect.
  // Must be greater than zero (to avoid the possibility of generating
  // NaN's).
  BaseFloat singular_value_relative_floor;


  // Floor for (spherical) variances; will be passed to class GaussianEstimator
  // when estimating means and variances.
  BaseFloat variance_floor;

  // A value in the range [0, 1] which dictates to what extent the variances are
  // shared.  0 means not shared at all, 1 means completely shared.  Shared
  // means the variance is a weighted average of variances, weighted by count of
  // that class.  This is consumed by class GaussianEstimator.
  BaseFloat variance_sharing_weight;

  // A count value of 'fake' counts that we add to the stats G, K and lambda
  // during estimation, namely:
  //   lambda += smoothing_count
  //   K += smoothing_count * smoothing_between_class_factor * I
  //   G += smoothing_count * I.
  // Interpretable as a number of frames.  This prevents things going crazy
  // when the amount of data is small.
  BaseFloat smoothing_count;

  // A factor that says how large the assumed between-class covariance matrix
  // is, relative to the within-class covariance matrix.  Should be >= 0.  In
  // the limit as it approaches zero, the smoothing will only penalize scaling
  // of the space, but not rotations.  This is likely not a good thing, so a
  // value greater than zero will probably be desired.
  BaseFloat smoothing_between_class_factor;

  FmllrEstimatorOptions():
      singular_value_relative_floor(0.001),
      variance_floor(0.0001),
      variance_sharing_weight(0.1),
      smoothing_count(0.0),
      smoothing_between_class_factor(0.25) { }

  void Check() {
    KALDI_ASSERT(singular_value_relative_floor > 0.0 &&
                 singular_value_relative_floor < 0.1 &&
                 (variance_floor > 0.0  || variance_sharing_weight > 0.0) &&
                 variance_floor >= 0.0 &&
                 variance_sharing_weight >= 0.0 &&
                 variance_sharing_weight <= 1.0);
  }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // This will set any options in this class that it can find in 'config_line'.
  void ReadFromConfig(ConfigLine *config_line);

};


/**
   Class CoreFmllrEstimator takes care of the core parts of the fMLLR estimation:
   with reference to the notation in
   http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf,
   it accepts the statistics G and K and the count gamma, and it
   computes the fMLLR transform matrix A, and allows you to backprop through
   that computation.  The reason why we have broken it out as its own class,
   is for testability and to limit the complexity of any one class.

   The end-user may want to use class FmllrEstimator instead.

 */
class CoreFmllrEstimator {
 public:
  /**
     Constructor.  Does not do any real work.  This class will store
     references/pointers to G, K and A, so you need to make sure that
     those quantities exist for the lifetime of this object.

       @param [in] opts  Options class; see its definition for details.  Will be copied
                      in the constructor.
       @param [in]  gamma  The total data-count (often this will be the number of frames).
       @param [in]  G  A symmetric matrix containing the quadratic
                       stats for estimating A.  This the sum of outer products
                       of the input features, after mean subtraction, and
                       weighted by the inverse-variance factor s_i.  Must be
                       positive definite for this computation to be well
                       defined.
       @param [in] K   A matrix containing the linear stats for estimating A.
                       This is a sum of outer products of the means with the
                       input features, with mean subtraction and inverse-variance
                       weighting.  Must not have more than one zero singular value
                       for this computation to be well defined.
       @param [in] A   We mark this as an input parameter but it is the location
                       where the output of this computation will be placed when
                       you call Forward().  May be undefined (e.g., NaN) on
                       entry.  You must not change the value of A between
                       calling Forward() and calling Backward().
   */
  CoreFmllrEstimator(const FmllrEstimatorOptions &opts,
                     BaseFloat gamma,
                     const MatrixBase<BaseFloat> &G,
                     const MatrixBase<BaseFloat> &K,
                     MatrixBase<BaseFloat> *A);

  /**
     Does the forward pass of estimation.  Writes to the location
     'A' that was passed to the constructor.

     Returns the objective-function improvement per frame, as compared
     with what the objective-function would be with unit A.  This is not
     normalized by the number of frames.
  */
  BaseFloat Forward();

  /**
     Does the backward pass.  Note: it is permissible to call
     Backward() any number of times, it does not have to be called
     exactly once.

       @param [in] A_deriv  The derivative of the objective
           function (say, f) w.r.t. the output A (which was passed as a
           pointer to the constructor).
       @param [out] G_deriv  A pointer to a location where the
           derivative df/dG will be written.  Will be added to, so
           should contain zero (or some other defined value)
           at input.
       @param [out] K_deriv  A pointer to a location where the
           derivative df/dK will be written (so the i,j'th
           element is the derivative w.r.t. the i,j'th element
           of the input matrix K.
  */
  void Backward(const MatrixBase<BaseFloat> &A_deriv,
                Matrix<BaseFloat> *G_deriv,
                Matrix<BaseFloat> *K_deriv);

 private:
  // Computes H = G^{-0.5}
  void ComputeH();
  // Compute L = K H
  void ComputeL();
  // Compute B = F(L), where F is the
  // function that takes the singular values of L, puts them through the function
  // f(lamba) = (lambda + sqrt(lambda^2 + 4 gamma)) / 2.
  void ComputeB();
  // Computes A = B H.
  void ComputeA();


  // Backprops through the operation "A = B H".  B_deriv and H_deriv
  // must be free of NaN and inf on entry.
  void BackpropA(const MatrixBase<BaseFloat> &A_deriv,
                 MatrixBase<BaseFloat> *B_deriv,
                 MatrixBase<BaseFloat> *H_deriv);

  // Backprops through the function "L = K H"..
  // K_deriv must be free of NaN and inf on entry, but otherwise
  // its value is ignored.  H_deriv is added to by this function.
  void BackpropL(const MatrixBase<BaseFloat> &L_deriv,
                 MatrixBase<BaseFloat> *K_deriv,
                 MatrixBase<BaseFloat> *H_deriv);

  // returns the objective-function change (vs. A being the unit matrix) from
  // this estimation.
  BaseFloat ComputeObjfChange();

  FmllrEstimatorOptions opts_;
  BaseFloat gamma_;
  const MatrixBase<BaseFloat> &G_;
  const MatrixBase<BaseFloat> &K_;
  MatrixBase<BaseFloat> *A_;

  // H = G^{-0.5} is symmetric.
  Matrix<BaseFloat> H_;
  // L = K H.
  Matrix<BaseFloat> L_;
  // B = F(L) is the result of applying SvdRescaler with
  // the function f(lambda) = ((lambda + sqrt(lambda^2 + 4 gamma)) / 2)
  Matrix<BaseFloat> B_;

  // Object that helps us to compute, and to backprop through the
  // computation of, H = G^{-0.5}.
  SvdRescaler G_rescaler_;

  // Object that helps us to compute, and to backprop through the computation
  // of: B = F(L), where F is the function that takes the singular values of L,
  // puts them through the function f(lamba) = (lambda + sqrt(lambda^2 + 4
  // gamma)) / 2.
  SvdRescaler L_rescaler_;

};



/**
   Class GaussianEstimator allows you to estimate means and (spherical) variances
   from features and posteriors, and to later backprop through that process if
   needed.

   It is intended for use during training of the neural net, for use on
   individual minibatches: it uses BaseFloat for the accumulators, which might
   lead to excessive roundoff if you had a large amount of data.  We'll later on
   create a separate mechanism for accumulating stats over all the data, given
   the full tree.

   The normal usage pattern would be:
      - Construct the object.
      - Call AccStats() for each sequence.
      - Call Estimate()
      - Call GetMeans() and GetVars() to obtain the means and vars, and do
        something with them, e.g. compute some kind of objective, from which
        you would obtain derivatives w.r.t. those means and vars.
      - Call SetOutputDerivs() to tell this class what those derivatives w.r.t.
        the means and vars are.
      - Call AccStatsBackward() for each sequence to propagate the derivatives
        back to the features that were used to estimate the means and vars.
 */
class GaussianEstimator {
 public:
  GaussianEstimator(int32 num_classes, int32 feature_dim);

  GaussianEstimator(const GaussianEstimator &other) = default;

  int32 NumClasses() const { return gamma_.Dim(); }

  int32 Dim() const;

  // Accumulate statistics (you can call this multiple times of needed).
  // It does: for each t, and for each pair (i, f) in post[t], accumulate stats
  // from feats.Row(t) with class i and weight f.
  // May not be called after Estimate() is called.
  //
  //   @param [in] feats   The input features, of dimension
  //                       num-frames by feature-dimension
  //   @param [in] post    The posteriors, which can be thought of as a
  //                       vector<vector<pair<int32,BaseFloat> > >.
  //                       Its size() must equal feats.NumRows().
  void AccStats(const MatrixBase<BaseFloat> &feats,
                const SubPosterior &post);

  // You call this once after calling AccStats() one or more times.
  // It estimates the model means and variances.
  // See the members 'variance_floor' and 'variance_sharing_weight'
  // of the options class.
  void Estimate(const FmllrEstimatorOptions &opts);

  // Returns true if Estimate() has previously been called, i.e. if
  // the means and variances have been computed.
  bool IsEstimated() const;

  // Returns the means, in a matrix of dimension num_classes by dim.  Must not
  // be called if ! IsEstimated().
  const MatrixBase<BaseFloat> &GetMeans() const { return mu_; }

  // Returns the 's' quantities, which are the scalar factors on the (spherical)
  // variances.  Must not be called if ! IsEstimated().  The
  // variance for class i will actually be s_i I, where s_i is an element of
  // this vector.
  const VectorBase<BaseFloat> &GetVars() const { return t_; }

  // You call this to add something the derivatives df/dmeans and df/dvars-- the
  // derivatives of the objective function f w.r.t. those quantities.  You might
  // call this once or several times.  Doing this allows you to backprop through
  // the estimation of the means and variances, back to the features.  This must
  // only be called after previously calling Estimate().  This function writes
  // to v_bar_ and m_bar_.
  void AddToOutputDerivs(const MatrixBase<BaseFloat> &mean_derivs,
                         const VectorBase<BaseFloat> &var_derivs);


  // This function, which must only be called after AddToOutputDerivs() has been
  // called at least once, propagates the derivative back to the features.  For
  // purposes of this backpropagation, the posteriors are treated as constants.
  //      @param [in] feats   The features, which must be the same
  //                          as you provided to one of the calls to
  //                          AccStats().  dimension is num-frames by
  //                          feature-dimension.
  //      @param [in] post    The posteriors, as provided to AccStats().
  //                          Its size() must equal feats.NumRows().
  //      @param [in,out] feats_deriv  The derivative of the objective
  //                          function w.r.t. the input features.
  //                          This function will *add to* feats_deriv,
  //                          so it must have a well-defined value on
  //                          entry.
  void AccStatsBackward(const MatrixBase<BaseFloat> &feats,
                        const SubPosterior &post,
                        const MatrixBase<BaseFloat> *feats_deriv);


  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Adds any statistics in gamma_, m_ and v_ from 'other' to *this.
  // Used when summing adaptation-model statistics over multiple
  // jobs.  Requires that '*this' and 'other' have identical
  // structure.
  void Add(const GaussianEstimator &other);

 private:
  /*
    Notes on implementation of GaussianEstimator.
    Using Latex notation.

     We are estimating means \mu_i and variance-factors s_i (these
     are scales on unit variances).  Later we'll apply a kind of
     interpolation with the global average variance, controlled
     by variance_sharing_weight_, and we'll call the variances that
     we finally output t_i.

     We formulate the sufficient statistics as:
      the counts \gamma_i, the mean stats m_i and the (scalar)
      variance stats v_i:

      \gamma_i = \sum_t \gamma_{t,i}
           m_i = \sum_t \gamma_{t,i} x_t
           v_i = \sum_t \gamma_{t,i} x_t^T x_t
     The estimation procedure is:
        \mu_i = \frac{m_i}{\gamma_i}, or 0 if \gamma_i is 0.
          s_i = variance_floor if \gamma_i = 0, else:
                max(variance_floor, (v_i/\gamma_i - \mu_i^T \mu_i) / dim)
         where dim is the feature dimension; and another form more convenient for backprop:
              = variance_floor if \gamma_i = 0, else:
                max(variance_floor, v_i/(dim * \gamma_i) - m_i^T m_i / (dim * \gamma_i^2))

     We write \bar{foo} for a derivative of the objective function w.r.t. foo.
     We are provided by the user with with \bar{\mu}_i and \bar{s}_i, when they
     call SetOutputDerivs(); and we aim to compute \bar{m}_i and \bar{v}_i, which
     are the derivs w.r.t. the raw statistics.  This is done as follows:
       \bar{m}_i = 0 if \gamma_i is 0, otherwise:
                     \frac{\bar{\mu}_i}{\gamma_i} - (\frac{2\bar{s}_i m_i}{dim \gamma_i^2}
                                                     if s_i > variance_floor, else 0)
                 = or 0 if \gamma_i is 0, otherwise:
                     \frac{\bar{\mu}_i}{\gamma_i} - (\frac{2\bar{s}_i \mu_i}{dim \gamma_i}
                                                     if s_i > variance_floor, else 0)
       \bar{v}_i = 0 if \gamma_i is 0 or s_i equals variance_floor, otherwise:
                     \frac{\bar{s}_i}{dim * \gamma_i}
       \bar{x}_t = \sum_i \gamma_{t,i} (\bar{m}_i + 2\bar{v}_i x_t)


    If 'variance_sharing_weight' != 0.0, then we need to modify the above.
    Let the variance-floored version of the variance be t_i.
    Write variance_sharing_weight as f (with 0 <= f <= 1), and let
        \gamma = \sum_i \gamma_i.
    Define the weighted-average variance:
         s  = \sum_i  \frac{\gamma_i}{\gamma} s_i
    and the partly-shared output variance is:
         t_i  = (1-f) s_i +  f s.
    For the backprop: If the user supplies derivatives \bar{t}_i, then:
          \bar{s} = f \sum_i \bar{t}_i
        \bar{s}_i = (1-f) \bar{t}_i  + \frac{\gamma_i}{\gamma} \bar{s}.
   */


  // gamma_, of dimension num_classes, contains the raw count statistics \gamma_i.
  // It's added to when you call AccStats().
  Vector<BaseFloat> gamma_;
  // m_ is the raw mean statistics (feature times soft-count); it's of dimension
  // num_classes by feat_dim.
  Matrix<BaseFloat> m_;
  // v_ is the raw variance statistics (inner-product-of-feature times soft-count);
  // it's of dimension num_classes.
  Vector<BaseFloat> v_;

  // variance_floor_ and variance_sharing_weight_ are copies of the
  // corresponding variables in class FmllrEstimatorOptions; they are set when
  // Estimate() is called.  They are temporaries, not permanent members.
  BaseFloat variance_floor_;
  BaseFloat variance_sharing_weight_;

  // mu_ is the estimated means, which is set up when you call Estimate().
  Matrix<BaseFloat> mu_;
  // s_ is the variances, after flooring by variance_floor_ but before
  // applying variance_sharing_weight_.
  Vector<BaseFloat> s_;
  // t_ is the smoothed or maybe totally averaged-over-all-classes variances,
  // derived from t as specified by variance_sharing_weight_.
  Vector<BaseFloat> t_;

  // v_bar_, of dimension num_classes, contains \bar{v}_i.  It's only set up
  // after you call SetOutputDerivs().
  Vector<BaseFloat> v_bar_;
  // m_bar_, of dimension num_classes by feature_dim, contains \bar{m}_i.
  // It's only set up after you call SetOutputDerivs().
  Matrix<BaseFloat> m_bar_;


};



/**
   Class FmllrEstimator encapsulates the whole of the fMLLR computation- for
   a single speaker.  See
     http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf
   for a description of what is being implemented here.

   This class is suitable for use in training, where you want to backprop
   through the computation; and also in test time (but not for the online
   scenario; we may later rewrite a version that's optimized for that, or modify
   this class to handle that).

   This class would normally be used as follows:
     - Construct an instance of the class (probably for a particular speaker on
       a particular minibatch).

   Then, either:

     - Call AccStats() one or more times.
     - Call Estimate().
     - Call AdaptFeatures() one or more times to get the output features.
        - Do something with those output features that (if you are training)
          gives you some kind of objective-function derivative w.r.t. those
         features.  Then if you are training, do what's below:
     - Call AdaptFeaturesBackward() one or more times to get part of the
       derivative w.r.t. the input features.  Note: the calls to AdaptFeatures()
       and AdaptFeaturesBackward() may be interleaved, since the call to
       AdaptFeatures() does not modify the object.
     - Call EstimateBackward()
     - Call AccStatsBackward() one or more times to get the part of the
       derivative w.r.t. the input features that comes from the effect
       on the transform itself.
     - Make use of the calls GetMeanDeriv() and GetVarDeriv() to
       account for the effect of the features on the class means and
       variances (these will be passed to class GaussianEstimator,
       and eventually to the features).

   Or: if there is only one training sequence, you can use the
   simplified interface:  after calling the constructor,

      - call ForwardCombined()
      - call BackwardCombined()
      - Make use of the calls GetMeanDeriv() and GetVarDeriv() to
        account for the effect of the features on the class means and
        variances, with the help of class GaussianEstimator.
*/
class FmllrEstimator {
 public:
  /**
     Constructor.
     @param [in] opts   Options class.  This class makes a copy.
     @param [in] mu     Class means, probably as output by class
                        GaussianEstimator.  This class maintains a
                        reference to this object, so you should ensure
                        that it exists for the lifetime of this object.
     @param [in] s      Scaling factors for spherical class
                        variances, probably as output by class
                        GaussianEstimator.  As with mu, we store
                        a reference to it, so don't destroy or
                        change it as long as this class instance exists.
  */
  FmllrEstimator(const FmllrEstimatorOptions &opts,
                 const MatrixBase<BaseFloat> &mu,
                 const VectorBase<BaseFloat> &s);


  /**
     Accumulate statistics to estimate the fMLLR transform.
       @param [in] feats  The feature matrix.  A row of it would be called
                       x_t in the writeup in
                       http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf.
       @param [in] post  The posteriors.  post.size() must equal feats.NumRows().
                       Each element of post is a list of pairs (i, p) where
                       i is the class label and p is the soft-count.
   */
  void AccStats(const MatrixBase<BaseFloat> &feats,
                const SubPosterior &post);


  /**
     Estimate the fMLLR transform parameters A and b.  Returns the
     objective-function improvement compared with A = I, b = 0, divided by the
     total count as returned by TotalCount().

     You are allowed to call this multiple times (e.g. call AccStats(), call
     Estimate(), call AccStats(), call Estimate() again).
  */
  BaseFloat Estimate();

  // Return true if Estimate() has previously been called.
  bool IsEstimated() const;

  /// Returns the total count of the posteriors accumulated so far.
  BaseFloat TotalCount() { return gamma_.Sum(); }

  /// Return the linear parameter matrix.  Adapted features are
  /// y_t = A x_t  +  b.  You won't necessarily need to
  /// call this, you can use ComputeAdaptedFeatures() intead.
  const MatrixBase<BaseFloat> &GetLinearParams() const { return A_; }

  /// Return the bias term b.
  const VectorBase<BaseFloat> &GetBiasParams() const { return b_; }

  /// Computes the adapted features y_t = A x_t + b.
  /// feats (x) and adapted_feats (y) must have the same dimension.  Must
  /// only be called after Estimate() has been called.
  /// 'adapted_feats' may contain NaN's on entry.
  void AdaptFeatures(const MatrixBase<BaseFloat> &feats,
                     MatrixBase<BaseFloat> *adapted_feats) const;

  /**
     This is the backward pass corresponding to the function AdaptFeatures().
     It propagates back only part of the derivative-- not including the part
     that's due to how the transform changes when the features change.  It
     also accumulates within this class instance the derivative w.r.t.
     A and b.  You are expected to later call EstimateBackward() and
     AccStatsBackward() to propagate the part of the derivative that comes from
     the effect on the transform, back to the input features.

     See also AccStatsBackward().
        @param [in]   feats    The features (x) that were the original input to
                               AdaptFeatures().
        @param [in]   adapted_feats_deriv  The derivative \bar{y} w.r.t. the output (y)
                               that was the result of calling AdaptFeatures().  Must
                               have the same size as feat.
        @param [in,out] feats_deriv   The derivative w.r.t. 'feats'; this function
                               *adds* to it.
   */
  void AdaptFeaturesBackward(const MatrixBase<BaseFloat> &feats,
                             const MatrixBase<BaseFloat> &adapted_feats_deriv,
                             MatrixBase<BaseFloat> *feats_deriv);

  /**
     This is the backward pass corresponding to Estimate().  You call this after
     calling AdaptFeaturesBackward() one or more times (which will accumulate
     the derivative w.r.t. A and B).  It backpropagates through the core
     estimation procedure of fMLLR, in preparation for you calling
     AccStatsBackward().
   */
  void EstimateBackward();


  // Returns the derivative w.r.t. the class means 'mu' that were supplied to the
  // constructor.  Must not be called until EstimateBackward() and
  // AccStatsBackward() have been called.
  const MatrixBase<BaseFloat> &GetMeanDeriv() const { return mu_bar_; }
  // Returns the derivative w.r.t. the variance factors 's' that were supplied
  // to the constructor.  Must not be called until EstimateBackward() and
  // AccStatsBackward() have been called.
  const VectorBase<BaseFloat> &GetVarDeriv() const { return s_bar_; }

  /**
     This is the backward pass corresponding to AccStats().  You call this after
     calling EstimateBackward().  It computes the part of the derivative w.r.t.
     'feats' that comes from the effect on the transform parameters.  You will
     normally have previously called AdaptFeaturesBackward() on these same
     features.
       @param [in] feats  The features as given to AccStats()
       @param [in] post   The posteriors as given to AccStats()
       @param [in,out] feats_deriv   This function *adds* to feats_deriv.
                          It adds the terms in \bar{x}_t that arise from
                          the derivative w.r.t. the transform parameters.  The
                          "direct" term \bar{x}_t = A^T \bar{y}_t will have
                          previously been added by AdaptFeaturesBackward().
   */
  void AccStatsBackward(const MatrixBase<BaseFloat> &feats,
                        const SubPosterior &post,
                        MatrixBase<BaseFloat> *feats_deriv);

  /**
     Combines AccStats(), Estimate() and AdaptFeatures() in one call;
     for use when there is only one sequence.  Returns the objective-function
     improvement (per soft-count).
        @param [in] feats  The features we're estimating the fMLLR parameters from
        @param [in] post   The posteriors corresponding to 'feats
        @param [out] adapted_feats   A matrix the same size as 'feats', to which
                           the adapted features will be written.  May contain
                           NaNs at entry.
   */
  BaseFloat ForwardCombined(const MatrixBase<BaseFloat> &feats,
                            const SubPosterior &post,
                            MatrixBase<BaseFloat> *adapted_feats);
  /**
     Combines AdaptFeaturesBackward(), EstimateBackward(), and
     AccStatsBackward(); for use when there is only one sequence.
     Note: 'feats_deriv' is *added* to so must be defined at entry.
  */
  void BackwardCombined(const MatrixBase<BaseFloat> &feats,
                        const SubPosterior &post,
                        const MatrixBase<BaseFloat> &adapted_feats_deriv,
                        MatrixBase<BaseFloat> *feats_deriv);

  ~FmllrEstimator();
 private:


  ///////////// Fixed quantities passed in in the constructor ///////////

  // The options.
  FmllrEstimatorOptions opts_;
  // The means.  A reference to an object owned elsewhere.
  const MatrixBase<BaseFloat> &mu_;
  // The variance factors (the variances are s_(i) times I).  A reference to an
  // object owned elsewhere.
  const VectorBase<BaseFloat> &s_;

  ///////////// Quantities that are accumulated in AccStats()  ///////////

  // Counts per class; dimension is num_classes.  Added to when AccStats() is
  // called.  gamma_(i) corresponds to \gamma_i in the write up; it's
  //   \gamma_i = \sum_t gamma_{t,i}
  Vector<BaseFloat> gamma_;

  // This contains one term in G_, namely:
  //   (\sum_t \hat{\gamma}_t x_t x_t^T )
  Matrix<BaseFloat> raw_G_;

  // This is of dimension num_classes by dim (same as mu_).  It contains
  // the weighted sums of the input data, for each class:
  //        z_i = \sum_t \gamma_{t,i} x_i.
  Matrix<BaseFloat> z_;


  /////////// Quantities that are computed when Estimate() is called  ////////


  // This contains
  //   G = (\sum_t \hat{\gamma}_t x_t x_t^T ) - \hat{\gamma} n n^T.
  // It is computed as raw_G_ - \hat{\gamma} n n^T.
  // We use two separate variables to make it easier to call Estimate()
  // more than once without things getting confused.
  Matrix<BaseFloat> G_;

  // gamma_hat_tot_ is the total of gamma_(i) / s_(i), i.e.
  //   \hat{\gamma} = \sum_i gamma_i / s_i.
  BaseFloat gamma_hat_tot_;

  // After Estimate() is called, this will be the quantity:
  //   n = \frac{1}{\hat{\gamma}} \sum_i (1/s_i) z_i
  Vector<BaseFloat> n_;

  // The weighted-average of the means:
  // m = \frac{1}{\hat{\gamma}} \sum_i (\gamma_i/s_i) \mu_i
  Vector<BaseFloat> m_;

  // This contains
  // K = (\sum_i (1/s_i) \mu_i z_i^T) - \hat{\gamma} m n^T
  Matrix<BaseFloat> K_;

  // The parameter matrix
  Matrix<BaseFloat> A_;
  // The offset term
  Vector<BaseFloat> b_;
  // The object we use to estimate A and b, and to backprop through that
  // process.
  CoreFmllrEstimator *estimator_;

  ////////// Quantities that are accumulated in AdaptFeaturesBackward() ////////

  // The derivative w.r.t. A.  This is set when AdaptFeaturesBackward() is called,
  // to:
  // \bar{A} = \sum_t \bar{y}_t x_t^T
  //   and then when EstimateBackward() is called, we add the term from the estimation
  //   of b, which is:
  // \bar{A} -=  \bar{b} n^T
  Matrix<BaseFloat> A_bar_;

  // The derivative w.r.t. b.  This is set when AdaptFeaturesBackward() is called,
  // to: \bar{b} = \sum_t \bar{y}_t.
  Vector<BaseFloat> b_bar_;

  ////////// Quantities that are computed in EstimateBackward() ////////

  // The derivative w.r.t. G; computed by 'estimator_'
  Matrix<BaseFloat> G_bar_;
  // The derivative w.r.t. K; computed by 'estimator_'.
  Matrix<BaseFloat> K_bar_;

  // The derivative w.r.t. n:
  // \bar{n} = -A^T \bar{b} - 2\hat{\gamma} \bar{G} n - \hat{\gamma} \bar{K}^T m
  Vector<BaseFloat> n_bar_;

  // The derivative w.r.t. m:
  // \bar{m} = \bar{b} - \hat{\gamma} \bar{K} n
  Vector<BaseFloat> m_bar_;

  // The derivative w.r.t the z_i quantities.  The i'th row is:
  //  \bar{z}_i =  (1/s_i) \bar{K}^T \mu_i  +  1/(s_i \hat{\gamma}) \bar{n}
  Matrix<BaseFloat> z_bar_;

  // gamma_hat_tot_bar_ is \bar{\hat{\gamma}} in the writeup;
  // it's:
  // \bar{\hat{\gamma}} = - n^T \bar{G} n - m^t \bar{K} n
  //                      - \frac{1}{\hat{\gamma}} (n^T \bar{n} + m^T \bar{m})
  BaseFloat gamma_hat_tot_bar_;

  // The i'th row contains the derivative w.r.t mu_i.
  // This is:
  // \bar{\mu}_i = (1/s_i) \bar{K} z_i + (\gamma_i / (s_i \hat{\gamma})) \bar{m}
  Matrix<BaseFloat> mu_bar_;

  //////////// Quantities that are written to in AccStatsBackward() ///////////

  // s_bar_(i) contains the derivative w.r.t the variance factor s_i,
  // which we write in the writeup as \bar{s}_i.
  // It is:
  //    \bar{s}_i  =  -(1 / s_i^2) * (
  //          \mu_i^T \bar{K} z_i  +  (1 / \hat{\gamma}) \z_i^T \bar{n}
  //       + (\gamma_i / \hat{\gamma}) \mu_i^T \bar{m}  + \gamma_i \bar{\hat{\gamma}}
  //       + \sum_t  \gamma_{t,i} \bar{\hat{\gamma}}_t )
  // where
  //  \bar{\hat{\gamma}}_t = x_t^T \bar{G} x_t  .
  // Note: we add all but the first terms during Estimate(), and only the one
  // with \sum_t in it in AccStatsBackward.
  Vector<BaseFloat> s_bar_;

  // There is another quantity that's updated by AccStatsBackward(), which is
  // \bar{x}_t, the derivative w.r.t. x_t.  AccStatsBackward() does not include
  // the term \bar{x}_t = A^T \bar{y}_t.  But it does include the rest of the
  // terms, doing:
  // \bar{x}_t  +=  2 \hat{\gamma}_t \bar{G} x_t
  //                 + \sum_i \gamma_{t,i} \bar{z}_i
  // There is no member variable for this; it's a temporary.

};


/* MeanOnlyTransformEstimator is like a highly simplified version of
   FmllrEstimator, where the transform is just y_t = x_t + b.
   There are class means but the variances are assumed to be all
   unit.  (This is equivalent to assuming that they are all identical
   with an arbitrary value; the value doesn't actually affect the
   learned offset so we assume they are unit).

   The equations involved are like an extremly simplified version
   of what we do in class FmllrEstimator, with m as a weighted
   average of the means and n as a weighted average of the input
   features.  The weights come from the posterior information you
   supply.

   This object has a similar interface to class FmllrEstimator.

   This class would normally be used as follows:
     - Construct an instance of the class (probably for a particular speaker on
       a particular minibatch).

   Then, either:

     - Call AccStats() one or more times.
     - Call Estimate().
     - Call AdaptFeatures() one or more times to get the output features.
        - Do something with those output features that (if you are training)
          gives you some kind of objective-function derivative w.r.t. those
         features.  Then if you are training, do what's below:
     - Call AdaptFeaturesBackward() one or more times to get part of the
       derivative w.r.t. the input features.  Note: the calls to AdaptFeatures()
       and AdaptFeaturesBackward() may be interleaved, since the call to
       AdaptFeatures() does not modify the object.
     - Call EstimateBackward()
     - Call AccStatsBackward() one or more times to get the part of the
       derivative w.r.t. the input features that comes from the effect
       on the transform itself.
     - Make use of the calls GetMeanDeriv() and GetVarDeriv() to
       account for the effect of the features on the class means and
       variances (these will be passed to class GaussianEstimator,
       and eventually to the features).

   Or: if there is only one training sequence, you can use the
   simplified interface:  after calling the constructor,

      - call ForwardCombined()
      - call BackwardCombined()
      - Make use of the call GetMeanDeriv() to account for the effect of the
        features on the class means and variances, with the help of class
        GaussianEstimator.
 */
class MeanOnlyTransformEstimator {
 public:
  /**
     Constructor.
     @param [in] mu     Class means, probably as output by class
                        GaussianEstimator.  This class maintains a
                        reference to this object, so you should ensure
                        that it exists for the lifetime of this object.
                        You can ignore the variances from class
                        GaussianEstimator; they are not used.
  */
  MeanOnlyTransformEstimator(const MatrixBase<BaseFloat> &mu);

  /**
     Accumulate statistics to estimate the fMLLR transform.
       @param [in] feats  The feature matrix.  A row of it would be called
                       x_t in the writeup in
                       http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf.
       @param [in] post  The posteriors.  post.size() must equal feats.NumRows().
                       Each element of post is a list of pairs (i, p) where
                       i is the class label and p is the soft-count.
   */
  void AccStats(const MatrixBase<BaseFloat> &feats,
                const SubPosterior &post);

  /**
     Estimate the parameter (the offset).  Requires the total count to be
     nonzero.  You are allowed to call this multiple times (e.g. call
     AccStats(), call Estimate(), call AccStats(), call Estimate() again).
  */
  void Estimate();

  // Returns true if Estimate() has previously been called.
  bool IsEstimated() const;

  BaseFloat TotalCount() { return gamma_.Sum(); }

  /// Return the bias term b.
  const VectorBase<BaseFloat> &GetOffset() const { return offset_; }

  /// Computes the adapted features y_t = x_t + b.
  /// feats (x) and adapted_feats (y) must have the same dimension.  Must
  /// only be called after Estimate() has been called.
  /// 'adapted_feats' may contain NaN's on entry.
  void AdaptFeatures(const MatrixBase<BaseFloat> &feats,
                     MatrixBase<BaseFloat> *adapted_feats) const;


  /**
     This is the backward pass corresponding to the function AdaptFeatures().
     It propagates back only part of the derivative-- not including the part
     that's due to how the offset changes when the features change.  It
     also accumulates within this class instance the derivative w.r.t. the
     offset.
     See also AccStatsBackward().

        @param [in]   feats    The features (x) that were the original input to
                               AdaptFeatures().
        @param [in]   adapted_feats_deriv  The derivative \bar{y} w.r.t. the output (y)
                               that was the result of calling AdaptFeatures().  Must
                               have the same size as feat.
        @param [in,out] feats_deriv   The derivative w.r.t. 'feats'; this function
                               *adds* to it.
   */
  void AdaptFeaturesBackward(const MatrixBase<BaseFloat> &feats,
                             const MatrixBase<BaseFloat> &adapted_feats_deriv,
                             MatrixBase<BaseFloat> *feats_deriv);

  /**
     Backward pass corresponding to Estimate().  Should be called after
     you've called AdaptFeatures() on all utterances.  Computes the
     derivatives w.r.t. the mean.  */
  void EstimateBackward();

  /**
     Returns the derivative w.r.t. the class means 'mu' that were supplied to
     the constructor.  Must not be called until EstimateBackward() has been
     called.  */
  const MatrixBase<BaseFloat> &GetMeanDeriv() const { return mu_bar_; }

  /**
     This is the backward pass corresponding to AccStats().  You call this after
     calling EstimateBackward().  It computes the part of the derivative w.r.t.
     'feats' that comes from the effect on the transform parameters.  You will
     normally have previously called AdaptFeaturesBackward() on these same
     features.
       @param [in] feats  The features as given to AccStats()
       @param [in,out] feats_deriv   This function *adds* to feats_deriv.
                          It adds the terms in \bar{x}_t that arise from
                          the derivative w.r.t. the offset b.
   */
  void AccStatsBackward(const MatrixBase<BaseFloat> &feats,
                        const SubPosterior &post,
                        MatrixBase<BaseFloat> *feats_deriv);


  /**
     Combines AccStats(), Estimate() and AdaptFeatures() in one call;
     for use when there is only one sequence.
        @param [in] feats  The features we're estimating the fMLLR parameters from
        @param [in] post   The posteriors corresponding to 'feats
        @param [out] adapted_feats   A matrix the same size as 'feats', to which
                           the adapted features will be written.  May contain
                           NaNs at entry.
   */
  void ForwardCombined(const MatrixBase<BaseFloat> &feats,
                       const SubPosterior &post,
                       MatrixBase<BaseFloat> *adapted_feats);
  /**
     Combines AdaptFeaturesBackward(), EstimateBackward(), and
     AccStatsBackward(); for use when there is only one sequence.
     Note: 'feats_deriv' is *added* to so must be defined at entry.
  */
  void BackwardCombined(const MatrixBase<BaseFloat> &feats,
                        const SubPosterior &post,
                        const MatrixBase<BaseFloat> &adapted_feats_deriv,
                        MatrixBase<BaseFloat> *feats_deriv);

 private:
  // The means, one row per class.  A reference to an object owned elsewhere.
  const MatrixBase<BaseFloat> &mu_;

  // The counts per class
  Vector<double> gamma_;
  // The total of the input features, weighted by total posterior.
  Vector<double> input_sum_;

  // The offset.
  Vector<BaseFloat> offset_;

  // The total of the derivative w.r.t. the output.
  Vector<double> output_deriv_sum_;

  // The derivative w.r.t. each row of the input features-- i.e. the part of the
  // derivative that comes from the effect via the offset.  This equals
  //  (-1 / total-count) * output_deriv_sum_.
  Vector<BaseFloat> x_deriv_;

  // The derivative w.r.t. mu:
  //    (1/gamma_tot) gamma_ . output_deriv_sum_^T.
  Matrix<BaseFloat> mu_bar_;
};


} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_DIFFERENTIABLE_FMLLR_H_
