// nnet2/train-nnet-perturbed.h

// Copyright 2012-2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_TRAIN_NNET_PERTURBED_H_
#define KALDI_NNET2_TRAIN_NNET_PERTURBED_H_

#include "nnet2/nnet-nnet.h"
#include "nnet2/nnet-example.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

/**
   @file

   This file was modified from train-nnet.h in order to implement an idea
   about perturbing the training examples slightly, in a direction that's
   opposite to the gradient of the objective function w.r.t. those examples.
   It's a bit like the idea in "Intriguing properties of neural networks", the
   training method they mention, except they have a more complicated formulation
   with L-BFGS.  We can justify our idea by approximating the neural network
   plus objective-function evaluation as a linear function.

   Note: before doing this, we want to make sure the input features have a
   reasonable distribution, and our choice for this is to make the within-class
   covariance matrix unit.  [note: we don't have to normalize the mean to zero,
   this won't matter.]  Rather than explicitly transforming the features using
   a transform T, it turns out that we have to multiply the gradients by something
   like T T'.  We'll describe this later.

   Suppose the actual input features are x.  Typically we do frame splicing
   as part of the network, and it's more convenient to do the perturbation on
   the spliced features, so x may actually be the output of the network's
   first (splicing) layer.  Suppose the within-class covariance matrix of
   x is W.  If we do the Cholesky transform
     W = C C^T,
   then C^{-1} W C^{-T} = I, so if we define
     T =(def) C^{-1} and
   and transformed features
     \hat{x} =(def) T x
   then it's easy to show that the within-class covariance matrix of the
   transformed features \hat{x} would be I.

   The way we formulate the perturbed-feature thing is somewhat similar to the
   "Intriguing properties of neural networks" paper, except we're not in image
   recognition so no need to keep features in the range [0, 1].  Given a training
   example \hat{x}_t, we want to find a perturbed example
       \hat{x}'_t = \hat{x}_t + d_t
   that gives the worst possible loss-value, such that ||d_t|| <= D, where D is
   a scalar length parameter (e.g. D = 0.1), and ||.|| is the 2-norm.  This means
   that we want to perturb the training example in the most damaging way possible,
   given that it should not change by more than a certain amount.  Because we've
   normalized the within-class covariance we believe that using a normal 2-norm
   on d_t, rather than a more general form of inner-product, is suitable.

   Anyway, we make a simplifying assumption that the loss function for a particular
   sample is just a linear function of the input, and when we get to the space of
   \hat{x}, it just means we go a certain distance D down the gradient.  How we
   set a suitable value for D, we'll come to later.
   
   Suppose by backpropagating the
   derivative to x we get a derivative \nabla_t of the objective function (e.g. a
   log-probability) w.r.t. x_t.  Then we can get the derivative \hat{\nabla}_t of
   the objective function w.r.t. \hat{x}_t, by identifying
       x_t^T nabla_t = \hat{x}_t^T \hat{\nabla}_t
       x_t^T nabla_t = x_t^T T^T \hat{\nabla}_t
       x_t^T nabla_t = x_t^T T^T T^{-T} \nabla_t, since T^T T^{-T} = I.
       [note, ^T is transpose and ^{-T} is inverse-of-transpose.]
   so  \hat{\nabla}_t = T^{-T} \nabla_t.
   (this is not the formal way of getting these derivatives, it's just how I remember).
   Anyway, we now have
       \hat{x}'_t =(def) \hat{x}_t  - k_t T^{-T} \nabla_t
   where k_t is chosen to ensure that
                        k_t || T^{-T} \nabla_t ||_2 = D
      k_t sqrt( \nabla_t^T T^{-1} T^{-T} \nabla_t ) = D
   so
     k_t = D / sqrt(\nabla_t^T T^{-1} T^{-T} \nabla_t)
         = D / sqrt(\nabla_t^T C C^T \nabla_t)
         = D / sqrt(\nabla_t^T W \nabla_t)
   Now, we actually want the update in terms of the parameter x instead of \hat{x},
   so multiplying the definition of \hat{x}'_t above by T^{-1} on the left, we have:
       x'_t = x_t - k_t T^{-1} T^{-T} \nabla_t
            = x_t - k_t W \nabla_t
  (note: we can also use W \nabla_t for efficiently computing k_t).

  It will actually be more efficient to do this after the FixedAffineTransform
  layer that we used to "precondition" the features, so after the second layer
  of the input rather than the first.  All we need to do is to get the
  within-class covariance matrix W in that space (after the
  FixedAffineTransform) instead.  We'll use the name x for that space, and forget
  about the original input space.

  Next, we want to discuss how we'll set the constant D.  D is a proportion of
  the within-class covariance.  However, it's not clear a priori how to set
  this, or that we can tune it just once and then leave it fixed for other
  setups.  For one thing, if the input features contain a lot of "nuisance"
  dimension that are not very informative about the class, it may be necessary
  for D to be smaller (because hopefully the gradients will be small in those
  nuisance directions).  There is another issue that this whole method is
  intended to improve generalization, so we only want to use it strongly if
  generalization is actually a problem.  For example, if we have so much
  training data and so few parameters that we have no trouble generalizing, we
  might not want to apply this method too strongly.  Our method will be to set D
  in order to get, on average, a certain degradation which we'll call
  "target-objf-change" in the objective function per frame.  Each time we
  apply this perturbation to a minibatch, we'll see whether the degradation in
  objective is greater or less than "target-objf-change", and we'll change
  D accordingly.  We'll use a simple heuristic that D should change proportionally
  to the 0.5'th power of the ratio between the "target-objf-change" and the
  observed objective function change for this minibatch, but never by more than
  a factor of two.  Note: the only significance of 0.5 here is that 0.5 <= 1; a
  smaller number means slower changes in D, so it should change over about 2
  minibatches to the right number.   If this proves unstable, we'll change it.

  Next, it's not absolutely clear how we should set target-objf-change-- the
  value which determines how much objective-function degradation we want the
  perturbation to produce on average (per sample).  To put this in perspective,
  for speech tasks with small amounts of data (say, <30 hours) and a couple thousand
  classes
  we typically see objective values like: training-set -0.6 and valdiation-set -1.1.
  These are avearage log-probabilities per frame, of the correct class.
  The two numbers are quite different because there is substantial overtraining.  Note: for Karel's
  nnet1 setup, the difference is typically smaller, more like -0.8 vs. -1.0, as
  that setup monitors the validation-set objective and decreases the learning rate
  when it starts to degrade.  Now, for much larger training sets, we might
  see smaller differences in training-set versus validation-set objective function:
  for instance: say, -1.40 versus -1.45.  (For larger training sets the objectives tend
  to be more negative simply because we have more leaves).  We measure these values each
  iteration: see the files compute_prob_train.*.log and compute_prob_valid.*.log produced
  by the example scripts.   The reason why I discuss these values
  is that if the training-set and validation-set objective functions are very close, then
  it means that there is not much overtraining going on and we don't want to apply this
  method too strongly; on the other hand, if they are very different, it means we are
  overtraining badly and we may want to apply this method more.

  So we plan to set target-objf-change to the following value, at the script level:

   target-objf-change = target-multiplier * (training-objf - validation-objf))

  (e.g. target-multiplier = 1.0).
  Note that if target-objf-change is less than a specified min-target-objf-change
  (e.g. 0.1) then we won't apply the perturbed training at all, which will save
  time.  The method is intended to help generalization, and if we're generalizing
  well then we don't need to apply it.
  The training and validation objective functions are computed over
  different (randomly chosen) sets, each with about 3000 samples, and it can
  sometimes happen that the validation objective function can be better than the
  training set objective function.  Also, the validation set is sampled from a
  held-out subset of 300 utterances by default; this is done out of a concern
  that the correlations within an utterance can be very high, so if we use the
  same utterances for training and validation, then the validation set is not
  really held-out.  But the smallish number (300) of validation utterances
  increases the randomness in the training and validation objectives.
*/



struct NnetPerturbedTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  // target_objf_change will be set from the command line to a value >0.0.
  BaseFloat target_objf_change;
  BaseFloat initial_d;
  // tune_d_power is not configurable from the command line.
  BaseFloat tune_d_power;
  // max_d_factor is not configurable from the command line.
  BaseFloat max_d_factor;


  NnetPerturbedTrainerConfig(): minibatch_size(500),
                                minibatches_per_phase(50),
                                target_objf_change(0.1),
                                initial_d(0.05),
                                tune_d_power(0.5),
                                max_d_factor(2.0){ }
  
  void Register (OptionsItf *po) {
    po->Register("minibatch-size", &minibatch_size,
                 "Number of samples per minibatch of training data.");
    po->Register("minibatches-per-phase", &minibatches_per_phase,
                 "Number of minibatches to wait before printing training-set "
                 "objective.");
    po->Register("target-objf-change", &target_objf_change, "Target objective "
                 "function change from feature perturbation, used to set "
                 "feature distance parameter D");
    po->Register("initial-d", &initial_d, "Initial value of parameter D "
                 "It will ultimately be set according to --target-objf-change");
  }  
};


/// Class NnetPerturbedTrainer is as NnetSimpleTrainer but implements feature
/// perturbation; see the comment at the top of this file (\ref
/// train-nnet-perturbed.h) for more details.

class NnetPerturbedTrainer {
 public:
  NnetPerturbedTrainer(const NnetPerturbedTrainerConfig &config,
                       const SpMatrix<BaseFloat> &within_class_covar,
                       Nnet *nnet);
  
  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetExample &value);

  ~NnetPerturbedTrainer();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetPerturbedTrainer);
  
  void TrainOneMinibatch();

  // This function initializes within_class_covar_ and num_layers_before_input_.
  // The input within_class_covar is the within-class covariance on the original
  // raw features, computed from LDA stats, but if this neural network has
  // a data-preconditioning layer of type FixedAffineComponent then we will
  // project the transform with that and treat the output of that transform
  // as the input x (this is more efficient).
  void InitWithinClassCovar(const SpMatrix<BaseFloat> &within_class_covar);

  void UpdateD(BaseFloat orig_objf_per_example,
               BaseFloat perturbed_objf_per_example);
  
  // The following function is called by TrainOneMinibatch() when we enter a new
  // phase.  A phase is just a certain number of epochs, and now matters only
  // for diagnostics (originally it meant something more).
  void BeginNewPhase(bool first_time);
  
  // Things we were given in the initializer:
  NnetPerturbedTrainerConfig config_;

  Nnet *nnet_; // the nnet we're training.

  // static information:
  // num_layers_before_input_ is the number of initial layers before what we
  // consider to be the input for this method: normally 2, for the splicing
  // layer and the (FixedAffineComponent) data-preconditioning layer.
  int32 num_layers_before_input_;
  // The within_class_covar_ variable below is the within-class covariance; if
  // we have a (FixedAffineComponent) data-preconditioning layer, we'd project
  // the within-class-covariance with that and store it as within_class_covar_.
  CuMatrix<BaseFloat> within_class_covar_;
  
  // State information:
  int32 num_phases_;
  int32 minibatches_seen_this_phase_;
  std::vector<NnetExample> buffer_;

  double logprob_this_phase_; // Needed for accumulating train log-prob on each phase.
  double logprob_perturbed_this_phase_;  // same for perturbed log-prob
  double weight_this_phase_; // count corresponding to the above.
  
  double logprob_total_;
  double logprob_perturbed_total_;
  double weight_total_;

  BaseFloat D_;  // The distance factor D.
};




/// This function computes the objective function and either updates the model
/// or adds to parameter gradients.  It returns the cross-entropy objective
/// function summed over all samples (normalize this by dividing by
/// TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for
/// a class NnetPerturbedUpdater that's defined in train-nnet-perturbed.cc, but we
/// don't want to expose that complexity at this level.
/// All these examples will be treated as one minibatch.
///
/// D is the distance factor that determines how much to perturb examples;
/// this is optimized in outer-level code (see class NnetPerturbedTrainer).
/// num_layers_before_input determines how many layers to skip before we find
/// the activation that we regard as the input x to the network, for purposes
/// of this method (e.g. we might skip over the splicing layer and a layer
/// that preconditions the input).
/// within_class_covar (actually a symmetric matrix, but represented as CuMatrix),
/// is the within-class covariance of the features, measured at that level,
/// which ultimately will be derived from LDA stats on the data.

void DoBackpropPerturbed(const Nnet &nnet,
                         int32 num_layers_before_input,
                         const CuMatrix<BaseFloat> &within_class_covar,
                         BaseFloat D,
                         const std::vector<NnetExample> &examples,
                         Nnet *nnet_to_update,
                         double *tot_objf_orig,
                         double *tot_objf_perturbed);



/// This function is similar to "DoBackpropParallel" as declared in
/// nnet-update-parallel.h, but supports "perturbed" training.  It's intended
/// for multi-threaded CPU-based training.  The number of threads will be
/// set to g_num_threads.
/// within_class_covar is the within-class covariance after any splicing
/// but before preconditioning, as needed for the LDA computation.
/// All pointer arguments must be non-NULL.
void DoBackpropPerturbedParallel(const NnetPerturbedTrainerConfig &config,
                                 const SpMatrix<BaseFloat> &within_class_covar,
                                 SequentialNnetExampleReader *example_reader,
                                 double *tot_objf_orig,
                                 double *tot_objf_perturbed,
                                 double *tot_weight,
                                 Nnet *nnet);


} // namespace nnet2
} // namespace kaldi

#endif
