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

#include "nnet-computation-graph.h"

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

/// Calls SetZero (with the given is_gradient parameter) on all updatable
/// components of the nnet.
void SetZero(bool is_gradient,
             Nnet *nnet);

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
///  It has an called "output" (other outputs are allowed but may be
///          ignored).
///  It has an input called "input", and possibly an extra input called
///    "ivector", but no other inputs.
///  There are probably some other properties that we really ought to
///  be checking, and we may add more later on.
bool IsSimpleNnet(const Nnet &nnet);

/// Zeroes the component stats in all nonlinear components in the nnet.
void ZeroComponentStats(Nnet *nnet);


/// ComputeNnetContext computes the left-context and right-context of a nnet.
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

/// Does *dest += alpha * src (affects nnet parameters and
/// stored stats).
void AddNnet(const Nnet &src, BaseFloat alpha, Nnet *dest);

/// Does *dest += alpha * src for updatable components (affect nnet parameters),
/// and *dest += scale * src for other components (affect stored stats).
/// Here, alphas is a vector of size equal to the number of updatable components
void AddNnetComponents(const Nnet &src, const Vector<BaseFloat> &alphas,
                       BaseFloat scale, Nnet *dest);

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

/// Convert all components of type RepeatedAffineComponent or
/// NaturalGradientRepeatedAffineComponent to BlockAffineComponent in nnet.
void ConvertRepeatedToBlockAffine(Nnet *nnet);

/// This function returns various info about the neural net.
/// If the nnet satisfied IsSimpleNnet(nnet), the info includes "left-context=5\nright-context=3\n...".  The info includes
/// the output of nnet.Info().
/// This is modeled after the info that AmNnetSimple returns in its
/// Info() function (we need this in the CTC code).
std::string NnetInfo(const Nnet &nnet);

/// This function sets the dropout proportion in all dropout component to 
/// dropout_proportion value.
void SetDropoutProportion(BaseFloat dropout_proportion, Nnet *nnet);

/// This function finds a list of components that are never used, and outputs
/// the integer comopnent indexes (you can use these to index
/// nnet.GetComponentNames() to get their names).
void FindOrphanComponents(const Nnet &nnet, std::vector<int32> *components);

/// This function finds a list of nodes that are never used to compute any
/// output, and outputs the integer node indexes (you can use these to index
/// nnet.GetNodeNames() to get their names).
void FindOrphanNodes(const Nnet &nnet, std::vector<int32> *nodes);


/**
   ReadEditConfig() reads a file with a similar-looking format to the config file
   read by Nnet::ReadConfig(), but this consists of a sequence of operations to
   perform on an existing network, mostly modifying components.  It's one
   "directive" (i.e. command) per line.

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
       Sets the learning rate for any updatable nodes matching the name pattern.

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
       Sets the dropout rates for any components of type DropoutComponent whose
       names match the given <name-pattern> (e.g. lstm*).  <name-pattern> defaults to "*".
   \endverbatim
*/
void ReadEditConfig(std::istream &config_file, Nnet *nnet);


} // namespace nnet3
} // namespace kaldi

#endif
