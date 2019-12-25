// pybind/chain/chain_training_pybind.cc

// Copyright 2019   Microsoft Corporation (author: Xingyu Na)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "chain/chain_pybind.h"

#include "chain/chain-training.h"

using namespace kaldi::chain;

void pybind_chain_training(py::module& m) {
  py::class_<ChainTrainingOptions>(m, "ChainTrainingOptions")
    .def(py::init<>())
    .def_readwrite("l2_regularize", &ChainTrainingOptions::l2_regularize,
                   "l2 regularization constant on the 'chain' output; the actual term added to"
                   "the objf will be -0.5 times this constant times the squared l2 norm."
                   "(squared so it's additive across the dimensions).  e.g. try 0.0005.")
    .def_readwrite("out_of_range_regularize", &ChainTrainingOptions::out_of_range_regularize,
                   "This is similar to an l2 regularization constant (like l2-regularize) but"
                   "applied on the part of the nnet output matrix that exceeds the range"
                   "[-30,30]... this is necessary to avoid things regularly going out of the"
                   "range that we can do exp() on, since the denominator computation is not in"
                   "log space and to avoid NaNs we limit the outputs to the range [-30,30].")
    .def_readwrite("leaky_hmm_coefficient", &ChainTrainingOptions::leaky_hmm_coefficient,
                   "Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from"
                   "each state to a special state with probability one, and then another"
                   "epsilon-transition from that special state to each state, with probability"
                   "leaky_hmm_coefficient times [initial-prob of destination state].  Imagine"
                   "we make two copies of each state prior to doing this, version A and version"
                   "B, with transition from A to B, so we don't have to consider epsilon loops-"
                   "or just imagine the coefficient is small enough that we can ignore the"
                   "epsilon loops."
                   "Note: we generally set leaky_hmm_coefficient to 0.1.")
    .def_readwrite("xent_regularize", &ChainTrainingOptions::xent_regularize,
                   "Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,"
                   "the network is expected to have an output named 'output-xent', which"
                   "should have a softmax as its final nonlinearity.");
}
