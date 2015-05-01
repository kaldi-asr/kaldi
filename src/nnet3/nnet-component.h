// nnet3/nnet-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang    
//                2014  Vijayaditya Peddinti
//                2014  Guoguo Chen

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

#ifndef KALDI_NNET3_NNET_COMPONENT_H_
#define KALDI_NNET3_NNET_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

// This is where we will put the actual Components, that inherit from the
// based-class Component defined in nnet-component-itf.h.  [Note: we may put the
// definitions of some more generic child classes of Component, such as
// UpdatableComponent, in nnet-component-itf.h.



// The rest of the comments here may be a bit outdated by now, and superseded by
// changes in the actual code.

// network =
//  (1) list of (real) components, with names.  These may be shared.  Info on
//      initialization (dimensions, etc.)  Just a list with no order.  May be
//      either SimpleComponent or GeneralComponent (GeneralComponent has a
//      more general interface).
//
//  (2) Abstract network structure (i.e. before seeing actual data).
//
//     List of input features with names and feature-dimensions (i.e. the number of
//       columns for each feature, but not the number of rows yet)
//     List of names of outputs, each with an InputDescriptor, for which see below (these may just be
//       the names of rel-component instances).
//
//     List of real-component instances.  [Note: a real component is something
//         that has a Component object; it might be of type SimpleComponent
//         (that has no interaction with indices like frame indices), or of type
//         GeneralComponent that may deal with frame indices.
//
//     For each real-component instance, a description of their input.
//     Call this InputDescriptor.
//
//     InputDescriptor is a list of SumDescriptor (interpreted as
//     appending them so the dimension gets larger). An SumDescriptor is a list
//     of inputs to be added together, all of the same dimension: it is a list of
//     ForwardingDescriptor.   A ForwardingDescriptor is
//     an object that translates indices, possibly from multiple input sources
//     (which in that case must all must be of the same dimension).
//     A ForwardingDescriptor may in general contain other ForwardingDescriptors,
//     so in general it represents a parse-tree type of expression (although
//     most of the common operators are unary).  The base-case of ForwardingDescriptor
//     is just the name of another network node.
//
//     There are several kinds of ForwardingDescriptor.
//     First it bifurcates into OneInputForwardingDescriptor and MultiInputForwardingDescriptor.
//     From OneInputForwardingDescriptor:
//       NodeForwardingDescriptor takes the input Index and turns it into a Cindex by
//         combining with the index of the node.
//       OffsetForwardingDescriptor applies a time-offset to a member ForwardingDescriptor.
//       We can have others later, e.g. one that sets all the t values to zero.
//     From MultiInputForwardingDescriptor:
//       SelectModuloForwardingDescriptor selects from different inputs; it takes a list of
//         input ForwardingDescriptors.
//
//
//   (3) concrete network structure.  We assume sufficient information
//   (SideInfo?) is passed in from outside that's sufficient to determine which
//   frames are required in things like LSTMs and aggregate-layers that have
//   arbitrary context.  In test-time we'll handle this with an AdaptationState
//   structure for each Component (attach one, non-const, to each Component, or at
//   least make it available during Propagate?)
//
//   Creating concrete network structure:
//    We declare the required output indices (for various output names, e.g.
//    just "posterior").
//    We declare the available input indices (assume these are just provided
//     regardless of need).
//
//    Define a Cindex as a pair (component-index, Index).
//    Keep going backward through the network computing required c-indices, until
//    either we fail due to something not available at input, or we have
//    everything we need.  Maintain a dependency graph where for each Cindex we
//    store the list of Cindexes it directly depends on.
//
//    OK, now we have a set of required c-indices.  
//    What order do we compute them in?  It will be based on what can be computed
//    first.  Each
//    Repeatedly do as follows:
//       What (component-index, Index) elements can be computed directly given
//    what we already have?  Select the component-index for which we can compute
//    the largest number of elements.  From this, create a CindexGroup, which
//    is a set of Cindex's all with the same component-index.  This corresponds
//    to the output of a real component.  Each CindexGroup will correspond to
//    one "real" Component's computation: call this one ConcreteComponentInstance.
//    It may depend on multiple previous
//    CindexGroups, we'll work this out later.  (note: the inputs of
//    the network will also get a CindexGroup index).
//
//    Now, for each each of the real components (and the input) the CindexGroup defines
//    the output and its order.  Now we create for each real component and the output,
//    a ConcreteInputDescriptor.
//   
//    A ConcreteInputDescriptor is a list of ConcreteSumDescriptor; the
//    ConcreteSumDescriptors are treated as different features to concatenate
//    together.  Each ConcreteSumDescriptor is a list of
//    ConcreteForwardingDescriptors, all of the same dimension, which will be
//    added together.  A ConcreteForwardingDescriptor, whether its underlying type is
//    OneInputForwardingDescriptor MultiInputForwardingDescriptor, is a list of
//    Cindex-- possibly from different components or from different CindexGroups.
//
//    The process of obtaining the ConcreteInputDescriptor is as follows.
//    We have at the output of the real component, a CindexGroup, i.e. a list
//    of Cindexes (incidentally these are unique and ordered).  We first map these
//    to a list of FrameIndexes at the input.  A SimpleComponent will just map them
//    one-to-one from the output Cindexes, but a GeneralComponent may do something
//    different.  Once we have the FrameIndexes on the input we do any mapping
//    necessary to get the ConcreteForwardingDescriptors.
//
//    OK, at this point of the computation we are going to work out where all the
//    data 'lives'.  Our goal is to get to a point where each output of a real
//    component, plus each input (i.e. each CindexGroup); and each ConcreteSum
//    Descriptor (i.e. each part of an input matrix to a component) gets a
//    SubMatrixIndex (identifying a sub-matrix), and these in turn each map to
//    offsets within a MatrixIndex (identifying an actual matrix).
//
//    For terminology: define an InputLocation as either the input of one of the
//    real components, or one of the designated outputs of the network; and
//    define an OutputLocation as either the output of one of the components,
//    or an input of the network,
//
//    We take it as given that each InputLocation has its own dedicated
//    location, i.e. a single matrix.  This simplifies our life w.r.t. the
//    backprop functions: they can just set the input value, rather than add to
//    them.  The only question becomes: for each output location, will it
//    have its own separate dedicated location, or will it be shared with
//    one of the ConcreteSumDescriptors? (i.e. a sub-matrix of one of the
//    inputs of another layer?).  For simplicity, at the moment we will
//    make the same sharing apply to the derivatives as to the parameters
//    themselves.
//
//    We can get arbitrarily sophisticated about how to do this, but I believe
//    the following rule (that says we are allowed to share) will cover most of
//    the common cases, and it's obviously correct: if the CindexGroup is the
//    same in both cases, and if none of the cindexes are used anywhere else,
//    and if the ConcreteForwardingDescriptor appears as the first member of its
//    ConcreteSumDescriptor [this is to avoid overwriting when we should be
//    adding], and if either (the component does not require its output in order
//    to backprop, or we're not doing backprop, or the ConcreteForwardingDescriptor
//    is the only member of its enclosing ConcreteSumDescriptor), then we can
//    share the location.
//
//    The next stage is to compile the actions of the computation.  We assume
//    when we start that the input already exists.
//
//    First compile the computation while ignoring any matrix resizing.  This
//    goes as follows:
//      For each ConcreteComponentInstance and for the outputs:
//         Do any steps needed to create the InputLocation [more details here.]
//         if it's a ConcreteComponentInstance: propagate.
//    
//    At this point, we let the user, from external code, call SetOutputDerivatives.
//    Then when they call Backprop(), a reverse sequence happens.
//
//    In general, it goes: for the outputs and for each ConcreteComponentInstance,
//      - if it's a ConcreteComponentInstance, do the backprop operation.fe  
//      - Propagate the derivative back (by adding) to each place where we added an
//        input from.
//
//    Later on we'll create a mechanism for selectively backpropagating the derivatives,
//    to avoid wasted computation.  For now, we'll just always do the backprop.
//    
//    The un-optimized computation will first create all of the forward matrices, then do the
//    forward computation, then create all of the backprop matrices, then do the
//    backward computation, then delete all the forward quantities, then delete all
//    the backprop quantities (except the derivatives at the input, if requested).
//
//    Note: we'll always give the empty matrix if a matrix is not needed: have a special
//    one for that.. we can make this an optimization step too.
//    
//    Then we optimize the matrix resizing times by creating matrices as late as possible
//    and destroying them as early as possible.





} // namespace nnet2
} // namespace kaldi


#endif

