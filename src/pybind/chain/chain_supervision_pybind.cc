// pybind/chain/chain_supervision_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
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

#include "chain/chain_pybind.h"

#include "chain/chain-supervision.h"

using namespace kaldi::chain;

void pybind_chain_supervision(py::module& m) {
  {
    using PyClass = Supervision;
    py::class_<PyClass>(m, "Supervision",
                        "struct Supervision is the fully-processed supervision "
                        "information for a whole utterance or (after "
                        "splitting) part of an utterance.  It contains the "
                        "time limits on phones encoded into the FST.")
        .def(py::init<>())
        .def(py::init<const PyClass&>(), py::arg("other"))
        .def("Swap", &PyClass::Swap)
        .def_readwrite("weight", &PyClass::weight,
                       "The weight of this example (will usually be 1.0).")
        .def_readwrite("num_sequences", &PyClass::num_sequences,
                       "num_sequences will be 1 if you create a Supervision "
                       "object from a single lattice or alignment, but if you "
                       "combine multiple Supevision objects the "
                       "'num_sequences' is the number of objects that were "
                       "combined (the FSTs get appended).")
        .def_readwrite("frames_per_sequence", &PyClass::frames_per_sequence,
                       "the number of frames in each sequence of appended "
                       "objects.  num_frames * num_sequences must equal the "
                       "path length of any path in the FST. Technically this "
                       "information is redundant with the FST, but it's "
                       "convenient to have it separately.")
        .def_readwrite("label_dim", &PyClass::label_dim,
                       "the maximum possible value of the labels in 'fst' "
                       "(which go from 1 to label_dim).  For fully-processed "
                       "examples this will equal the NumPdfs() in the "
                       "TransitionModel object, but for newer-style "
                       "'unconstrained' examples that have been output by "
                       "chain-get-supervision but not yet processed by "
                       "nnet3-chain-get-egs, it will be the NumTransitionIds() "
                       "of the TransitionModel object.")
        .def_readwrite(
            "fst", &PyClass::fst,
            "This is an epsilon-free unweighted acceptor that is sorted in "
            "increasing order of frame index (this implies it's topologically "
            "sorted but it's a stronger condition).  The labels will normally "
            "be pdf-ids plus one (to avoid epsilons, since pdf-ids are "
            "zero-based), but for newer-style 'unconstrained' examples that "
            "have been output by chain-get-supervision but not yet processed "
            "by nnet3-chain-get-egs, they will be transition-ids. Each "
            "successful path in 'fst' has exactly 'frames_per_sequence * "
            "num_sequences' arcs on it (first 'frames_per_sequence' arcs for "
            "the first sequence; then 'frames_per_sequence' arcs for the "
            "second sequence, and so on).")
        .def_readwrite(
            "e2e_fsts", &PyClass::e2e_fsts,
            "'e2e_fsts' may be set as an alternative to 'fst'.  These FSTs are "
            "used when the numerator computation will be done with 'full "
            "forward_backward' instead of constrained in time.  (The "
            "'constrained in time' fsts are how we described it in the "
            "original LF-MMI paper, where each phone can only occur at the "
            "same time it occurred in the lattice, extended by a tolerance)."
            "\n"
            "This 'e2e_fsts' is an array of FSTs, one per sequence, that are "
            "acceptors with (pdf_id + 1) on the labels, just like 'fst', but "
            "which are cyclic FSTs. Unlike with 'fst', it is not the case with "
            "'e2e_fsts' that each arc corresponds to a specific frame)."
            "\n"
            "There are two situations 'e2e_fsts' might be set. The first is in "
            "'end-to-end' training, where we train without a tree from a flat "
            "start.  The function responsible for creating this object in that "
            "case is TrainingGraphToSupervision(); to find out more about "
            "end-to-end training, see chain-generic-numerator.h The second "
            "situation is where we create the supervision from lattices, and "
            "split them into chunks using the time marks in the lattice, but "
            "then make a cyclic FST, and don't enforce the times on the "
            "lattice inside the chunk.  [Code location TBD].")
        .def_readwrite("alignment_pdfs", &PyClass::alignment_pdfs,
                       "This member is only set to a nonempty value if we are "
                       "creating 'unconstrained' egs.  These are egs that are "
                       "split into chunks using the lattice alignments, but "
                       "then within the chunks we remove the frame-level "
                       "constraints on which phones can appear when, and use "
                       "the 'e2e_fsts' member."
                       "\n"
                       "It is only required in order to accumulate the LDA "
                       "stats using `nnet3-chain-acc-lda-stats`, and it is not "
                       "merged by nnet3-chain-merge-egs; it will only be "
                       "present for un-merged egs.")
        .def("__str__",
             [](const PyClass& sup) {
               std::ostringstream os;
               os << "weight: " << sup.weight << "\n"
                  << "num_sequences: " << sup.num_sequences << "\n"
                  << "frames_per_sequence: " << sup.frames_per_sequence << "\n"
                  << "label_dim: " << sup.label_dim << "\n";
               return os.str();
             })
        // TODO(fangjun): Check, Write and Read are not wrapped
        ;
  }
}
