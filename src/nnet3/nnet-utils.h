// nnet3/nnet-utils.h

// Copyright   2015  Johns Hopkins University (author: Daniel Povey)
//             2016  Daniel Galvez
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

#ifndef KALDI_NNET3_NNET_UTILS_H_
#define KALDI_NNET3_NNET_UTILS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-descriptor.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {


/// \file nnet3/nnet-utils.h
/// This file contains some miscellaneous functions dealing with class Nnet.

/// Given an nnet and a computation request, this function works out which
/// requested outputs in the computation request are computable; it outputs this
/// information as a vector "is_computable" indexed by the same indexes as
/// request.outputs.
/// It does this by executing some of the early stages of compilation.
void EvaluateComputationRequest(
    const Nnet &nnet,
    const ComputationRequest &request,
    std::vector<std::vector<bool> > *is_computable);


/// returns the number of output nodes of this nnet.
int32 NumOutputNodes(const Nnet &nnet);

/// returns the number of input nodes of this nnet.
int32 NumInputNodes(const Nnet &nnet);

/// Calls PerturbParams (with the given stddev) on all updatable components of
/// the nnet.
void PerturbParams(BaseFloat stddev,
                   Nnet *nnet);


/// Returns dot product between two networks of the same structure (calls the
/// DotProduct functions of the Updatable components and sums up the return
/// values).
BaseFloat DotProduct(const Nnet &nnet1,
                     const Nnet &nnet2);

/// Returns dot products between two networks of the same structure (calls the
/// DotProduct functions of the Updatable components and fill in the output
/// vector).
void ComponentDotProducts(const Nnet &nnet1,
                          const Nnet &nnet2,
                          VectorBase<BaseFloat> *dot_prod);

/// This function is for printing, to a string, a vector with one element per
/// updatable component of the nnet (e.g. the output of ComponentDotProducts),
/// in a human readable way, as [ component-name1:number1
/// component-name2:number2 ... ].
std::string PrintVectorPerUpdatableComponent(const Nnet &nnet,
                                             const VectorBase<BaseFloat> &vec);

/// This function returns true if the nnet has the following properties:
///  It has an output called "output" (other outputs are allowed but may be
///  ignored).
///  It has an input called "input", and possibly an extra input called
///    "ivector", but no other inputs.
///  There are probably some other properties that we really ought to
///  be checking, and we may add more later on.
bool IsSimpleNnet(const Nnet &nnet);

/// Zeroes the component stats in all nonlinear components in the nnet.
void ZeroComponentStats(Nnet *nnet);


/// ComputeSimpleNnetContext computes the left-context and right-context of a nnet.
/// The nnet must satisfy IsSimpleNnet(nnet).
///
/// It does this by constructing a ComputationRequest with a certain number of inputs
/// available, outputs can be computed..  It does the same after shifting the time
/// index of the output to all values 0, 1, ... n-1, where n is the output
/// of nnet.Modulus().   Then it returns the largest left context and the largest
/// right context that it infers from any of these computation requests.
void ComputeSimpleNnetContext(const Nnet &nnet,
                              int32 *left_context,
                              int32 *right_context);


/// Sets the underlying learning rate for all the components in the nnet to this
/// value.  this will get multiplied by the individual learning-rate-factors to
/// produce the actual learning rates.
void SetLearningRate(BaseFloat learning_rate,
                     Nnet *nnet);

/// Scales the nnet parameters and stats by this scale.
void ScaleNnet(BaseFloat scale, Nnet *nnet);

/// Sets nnet as gradient by Setting is_gradient_ to true and
/// learning_rate_ to 1 for each UpdatableComponent in nnet
void SetNnetAsGradient(Nnet *nnet);

/// Does *dest += alpha * src (affects nnet parameters and
/// stored stats).
void AddNnet(const Nnet &src, BaseFloat alpha, Nnet *dest);

/// Does *dest += alpha * src for updatable components (affects nnet parameters),
/// and *dest += scale * src for other components (affects stored stats).
/// Here, alphas is a vector of size equal to the number of updatable components
void AddNnetComponents(const Nnet &src, const Vector<BaseFloat> &alphas,
                       BaseFloat scale, Nnet *dest);

/// Returns true if 'nnet' has some kind of recurrency.
bool NnetIsRecurrent(const Nnet &nnet);

/// Returns the total of the number of parameters in the updatable components of
/// the nnet.
int32 NumParameters(const Nnet &src);

/// Copies the nnet parameters to *params, whose dimension must
/// be equal to NumParameters(src).
void VectorizeNnet(const Nnet &src,
                   VectorBase<BaseFloat> *params);


/// Copies the parameters from params to *dest.  the dimension of params must
/// be equal to NumParameters(*dest).
void UnVectorizeNnet(const VectorBase<BaseFloat> &params,
                     Nnet *dest);

/// Returns the number of updatable components in the nnet.
int32 NumUpdatableComponents(const Nnet &dest);

/// Controls if natural gradient will be updated
void FreezeNaturalGradient(bool freeze, Nnet *nnet);

/// Convert all components of type RepeatedAffineComponent or
/// NaturalGradientRepeatedAffineComponent to BlockAffineComponent in nnet.
void ConvertRepeatedToBlockAffine(Nnet *nnet);

/// This function returns various info about the neural net.
/// If the nnet satisfied IsSimpleNnet(nnet), the info includes "left-context=5\nright-context=3\n...".  The info includes
/// the output of nnet.Info().
/// This is modeled after the info that AmNnetSimple returns in its
/// Info() function (we need this in the CTC code).
std::string NnetInfo(const Nnet &nnet);

/// This function sets the dropout proportion in all dropout components to
/// dropout_proportion value.
void SetDropoutProportion(BaseFloat dropout_proportion, Nnet *nnet);


/// Returns true if nnet has at least one component of type BatchNormComponent.
bool HasBatchnorm(const Nnet &nnet);

/// This function affects only components of type BatchNormComponent.
/// It sets "test mode" on such components (if you call it with test_mode =
/// true, otherwise it would set normal mode, but this wouldn't be needed
/// often).  "test mode" means that instead of using statistics from the batch,
/// it does a deterministic normalization based on statistics stored at training
/// time.
void SetBatchnormTestMode(bool test_mode, Nnet *nnet);


/// This function zeros the stored component-level stats in the nnet using
/// ZeroComponentStats(), then recomputes them with the supplied egs.  It
/// affects batch-norm, for instance.  See also the version of RecomputeStats
/// declared in nnet-chain-diagnostics.h.
void RecomputeStats(const std::vector<NnetExample> &egs, Nnet *nnet);



/// This function affects components of child-classes of
/// RandomComponent.
/// It sets "test mode" on such components (if you call it with test_mode =
/// true, otherwise it would set normal mode, but this wouldn't be needed often).
/// "test mode" means that having a mask containing (1-dropout_prob) in all
/// elements.
void SetDropoutTestMode(bool test_mode, Nnet *nnet);

/**
  \brief  This function calls 'ResetGenerator()' on all components in 'nnet'
     that inherit from class RandomComponent.  It's used when you need
     to ensure consistency in things like dropout masks, across subsequent
     neural net evaluations.  You will likely want to call srand() before calling
     this.
*/
void ResetGenerators(Nnet *nnet);

/// This function finds a list of components that are never used, and outputs
/// the integer comopnent indexes (you can use these to index
/// nnet.GetComponentNames() to get their names).
void FindOrphanComponents(const Nnet &nnet, std::vector<int32> *components);

/// This function finds a list of nodes that are never used to compute any
/// output, and outputs the integer node indexes (you can use these to index
/// nnet.GetNodeNames() to get their names).
void FindOrphanNodes(const Nnet &nnet, std::vector<int32> *nodes);



/**
   Config class for the CollapseModel function.  This function
   is reponsible for collapsing together sequential components where
   doing so could make the test-time operation more efficient.
   For example, dropout components and batch-norm components that
   are in test mode can be combined with the next layer; and if there
   are successive affine components it may also be possible to
   combine these under some circumstances.

   It expects batch-norm components to be in test mode; you should probably call
   SetBatchnormTestMode() and SetDropoutTestMode() before CollapseModel().
 */
struct CollapseModelConfig {
  bool collapse_dropout;  // dropout then affine/conv.
  bool collapse_batchnorm;  // batchnorm then affine.
  bool collapse_affine;  // affine or fixed-affine then affine.
  bool collapse_scale;  // affine then fixed-scale.
  CollapseModelConfig(): collapse_dropout(true),
                         collapse_batchnorm(true),
                         collapse_affine(true),
                         collapse_scale(true) { }
};

/**
   This function modifies the neural net for efficiency, in a way that
   suitable to be done in test time.  For example, it tries to get
   rid of dropout, batchnorm and fixed-scale components, and to
   collapse subsequent affine components if doing so won't hurt
   speed.
 */
void CollapseModel(const CollapseModelConfig &config,
                   Nnet *nnet);

/**
   ReadEditConfig() reads a file with a similar-looking format to the config file
   read by Nnet::ReadConfig(), but this consists of a sequence of operations to
   perform on an existing network, mostly modifying components.  It's one
   "directive" (i.e. command) per line, but if supplying the options via
   the --edits option to programs like nnet3-am-copy, you can use a semicolon
   in place of the newline to separate commands.

   The following describes the allowed commands.  Note: all patterns are like
   UNIX globbing patterns where the only metacharacter is '*', representing zero
   or more characters.

  \verbatim
    convert-to-fixed-affine [name=<name-pattern>]
      Converts the given affine components to FixedAffineComponent which is not updatable.

    remove-orphan-nodes [remove-orphan-inputs=(true|false)]
      Removes orphan nodes (that are never used to compute anything).  Note:
      remove-orphan-inputs defaults to false.

    remove-orphan-components
      Removes orphan components (those that are never used by any node).

    remove-orphans [remove-orphan-inputs=(true|false)]
      The same as calling remove-orphan-nodes and then remove-orphan-components.

    set-learning-rate [name=<name-pattern>] learning-rate=<learning-rate>
       Sets the learning rate for any updatable components matching the name pattern.
       Note: this sets the 'underlying' learning rate, i.e. it will get
       multiplied by any 'learning-rate-factor' set in the components.

    set-learning-rate-factor [name=<name-pattern>] learning-rate-factor=<learning-rate-factor>
       Sets the learning rate factor for any updatable components matching the name pattern.

    rename-node old-name=<old-name> new-name=<new-name>
       Renames a node; this is a surface renaming that does not affect the structure
       (for structural changes, use the regular config file format, not the
       edits-config).  This is mostly useful for outputs, e.g. when doing
       multilingual experiments.

    remove-output-nodes name=<name-pattern>
       Removes a subset of output nodes, those matching the pattern.  You cannot
       remove internal nodes directly; instead you should use the command
       'remove-orphans'.

    set-dropout-proportion [name=<name-pattern>] proportion=<dropout-proportion>
       Sets the dropout rates for any components of type DropoutComponent,
       DropoutMaskComponent or GeneralDropoutComponent whose
       names match the given <name-pattern> (e.g. lstm*).  <name-pattern> defaults to "*".

    apply-svd name=<name-pattern> bottleneck-dim=<dim>
       Locates all components with names matching <name-pattern>, which are
       type AffineComponent or child classes thereof.  If <dim> is
       less than the minimum of the (input or output) dimension of the component,
       it does SVD on the components' parameters, retaining only the alrgest
       <dim> singular values, replacing these components with sequences of two
       components, of types LinearComponent and NaturalGradientAffineComponent.
       See also 'reduce-rank'.

    reduce-rank name=<name-pattern> rank=<dim>
       Locates all components with names matching <name-pattern>, which are
       type AffineComponent or child classes thereof.  Does SVD on the
       components' parameters, retaining only the largest <dim> singular values,
       and writes the reconstructed matrix back to the component.  See also
       'apply-svd', which structurally breaks the component into two pieces.

   \endverbatim
*/
void ReadEditConfig(std::istream &config_file, Nnet *nnet);

/**
   This function does the operation '*nnet += scale * delta_nnet', while
   respecting any max-parameter-change (max-param-change) specified in the
   updatable components, and also the global max-param-change specified as
   'max_param_change'.

   With max-changes taken into account, the operation of this function is
   equivalent to the following, although it's done more efficiently:

   \code
     Nnet temp_nnet(delta_nnet);
     ScaleNnet(1.0 / max_change_scale, &temp_nnet);
     [ Scale down parameters for each component of temp_nnet as needed so
     their Euclidean norms do not exceed their per-component max-changes ]
     [ Scale down temp_nnet as needed so its Euclidean norm does not exceed
       the global max-change ]
     ScaleNnet(max_change_scale, &temp_nnet);  // undo the previous scaling.
     AddNnet(temp_nnet, scale, nnet);
   \endcode

   @param [in] delta_nnet  The copy of '*nnet' neural network that contains
               the proposed change in parameters. Normally this will previously
               have been set to: (delta_nnet =
               parameter-derivative-on-current-minibatch *
               learning-rate per parameter), with any natural gradient applied
               as specified in the components; but this may be different if
               momentum or backstitch are used.
   @param [in] max_param_change  The global max-param-change specified on the
               command line (e.g. 2.0), which specifies the largest change
               allowed to '*nnet' in Euclidean norm.  If <= 0, no global
               max-param-change will be enforced, but any max-change values
               specified in the components will still be enforced; see
               UpdatableComponent::MaxChange(), and search for 'max-change' in
               the configs or nnet3-info output).
   @param [in] max_change_scale  This value, which will normally be 1.0, is used
               to scale all per-component max-change values and the global
               'max_param_change', before applying them (so we use
               'max_change_scale * uc->MaxChange()' as the per-component
               max-change, and 'max_change_scale * max_param_change' as the
               global max-change).
   @param [in] scale  This value, which will normally be 1.0, is a scaling
               factor used when adding to 'nnet', applied after any max-changes.
               It is provided for backstitch-related purposes.
   @param [in,out] nnet  The nnet which we add to.
   @param [out] num_max_change_per_component_applied  We add to the elements of
                   this the count for each per-component max-change.
   @param [out] num_max_change_global_applied  We to this the count for the
                   global max-change.
*/
bool UpdateNnetWithMaxChange(const Nnet &delta_nnet,
                             BaseFloat max_param_change,
                             BaseFloat max_change_scale,
                             BaseFloat scale, Nnet *nnet,
                             std::vector<int32> *
                             num_max_change_per_component_applied,
                             int32 *num_max_change_global_applied);


/**
   This function is used as part of the regular training workflow, prior to
   UpdateNnetWithMaxChange().

   For each updatable component c in the neural net, suppose it has a
   l2-regularization constant alpha set at the component level (see
   UpdatableComponent::L2Regularization()), and a learning-rate
   eta, then this function does (and this is not real code):

        delta_nnet->c -= 2.0 * l2_regularize_scale * alpha * eta * nnet.c

    The factor of -1.0 comes from the fact that we are maximizing, and we'd
    add the l2 regularization term (of the form ||\theta||_2^2, i.e. squared
    l2 norm) in the objective function with negative sign; the factor of 2.0
    comes from the derivative of the squared parameters.  The factor
    'l2_regularize_scale' is provided to this function, see below for an
    explanation.

    Note: the way we do it is a little bit approximate, due to the interaction
    with natural gradient.  The issue is that the regular gradients are
    multiplied by the inverse of the approximated, smoothed and factored inverse
    Fisher matrix, but the l2 gradients are not.  This means that what we're
    optimizing is not exactly the (regular objective plus the L2 term)-- we
    could view it as optimizing (regular objective plus the l2 term times the
    Fisher matrix)-- with the proviso that the Fisher matrix has been scaled in
    such a way that the amount of parameter change is not affected, so this is
    not an issue of affecting the overall strength of l2, just an issue of the
    direction-wise weighting.  In effect, the l2 term will be larger, relative
    to the gradient contribution, in directions where the Fisher matrix is
    large.  This is probably not ideal-- but it's hard to judge without
    experiments.  Anyway the l2 effect is small enough, and the Fisher matrix
    sufficiently smoothed with the identity, that I doubt this makes much of a
    difference.

    @param [in] nnet  The neural net that is being trained; expected
                      to be different from delta_nnet
    @param [in] l2_regularize_scale   A scale on the l2 regularization.
                      Usually this will be equal to the number of
                      distinct examples (e.g. the number of chunks of
                      speech-- more precisely, the number of distinct
                      'n' values) in the minibatch, but this is
                      multiplied by a configuration value
                      --l2-regularize-factor passed in from the command
                      line.  The reason for making l2 proportional to
                      the number of elements in the minibatch is that
                      we add the parameter gradients over the minibatch
                      (we don't average), so multiplying the l2 factor by the
                      number of elements in the minibatch is necessary to
                      make the amount of l2 vs. gradient contribution stay
                      the same when we vary the minibatch size.
                      The --l2-regularize-factor option is provided so that the
                      calling script can correct for the effects of
                      parallelization via model-averaging (we'd normally set
                      this to 1/num-parallel-jobs).
    @param [out] delta_nnet  The neural net containing the parameter
                      updates; this is a copy of 'nnet' that is used
                      for purposes of momentum and applying max-change
                      values.  This is what this code adds to.
 */
void ApplyL2Regularization(const Nnet &nnet,
                           BaseFloat l2_regularize_scale,
                           Nnet *delta_nnet);


/**
   This function scales the batchorm stats of any batchnorm components
   (components of type BatchNormComponent) in 'nnet' by the scale
   'batchnorm_stats_scale'.
 */
void ScaleBatchnormStats(BaseFloat batchnorm_stats_scale,
                         Nnet *nnet);


/**
   This function, to be called after processing every minibatch, is responsible
   for enforcing the orthogonality constraint for any components of type
   LinearComponent or inheriting from AffineComponent that have the
   "orthonormal-constraint" value set to a nonzero value.

   Technically what we are doing is constraining the parameter matrix M to be a
   "semi-orthogonal" matrix times a constant alpha.  That is: if num-rows >
   num-cols, this amounts to asserting that M M^T == alpha^2 I; otherwise, that
   M^T M == alpha^2 I.

   If, for a particular component, orthonormal-constraint > 0.0, then that value
   becomes the "alpha" mentioned above.  If orthonormal-constraint == 0.0, then
   nothing is done.  If orthonormal-constraint < 0.0, then it's like letting alpha
   "float", i.e. we try to make M closer to (any constant alpha) times a
   semi-orthogonal matrix.

   In order to make it efficient on GPU, it doesn't make it completely orthonormal,
   it just makes it closer to being orthonormal (times the 'orthonormal_constraint'
   value).  Over multiple iterations this rapidly makes it almost exactly orthonormal.
 */
void ConstrainOrthonormal(Nnet *nnet);

/** This utility function can be used to obtain the number of distinct 'n'
    values in a training example.  This is the number of examples
    (e.g. sequences) that have been combined into a single example.  (Actually
    it returns the (largest - smallest + 1) of 'n' values, and assumes they are
    consecutive).


    @param [in] vec    The vector of NnetIo objects from the training example
                       (NnetExample or NnetChainExample) for which we need the
                       number of 'n' values
    @param [in] exhaustive   If true, it will check exhaustively what largest
                        and smallest 'n' values are.  If 'false' it does it in a
                        fast way which will return the same answer as if
                        exhaustive == true for all the types of eg we currently
                        create (basically: correct if the last row of the input
                        or supervision matrices has the last-numbered 'n'
                        value), and will occasionally (randomly) do a test to
                        check that this is the same as if we called it with
                        'exhaustive=true'.
 */
int32 GetNumNvalues(const std::vector<NnetIo> &io_vec,
                    bool exhaustive);


} // namespace nnet3
} // namespace kaldi

#endif
