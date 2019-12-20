// pybind/nnet3/nnet_chain_example_pybind.cc

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

#include "nnet3/nnet_chain_example_pybind.h"

#include "nnet3/nnet-chain-example.h"
#include "util/kaldi_table_pybind.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace kaldi::chain;

void pybind_nnet_chain_example(py::module& m) {
  {
    using PyClass = NnetChainSupervision;
    py::class_<PyClass>(
        m, "NnetChainSupervision",
        "For regular setups we use struct 'NnetIo' as the output.  For the "
        "'chain' models, the output supervision is a little more complex as it "
        "involves a lattice and we need to do forward-backward, so we use a "
        "separate struct for it.  The 'output' name means that it pertains to "
        "the output of the network, as opposed to the features which pertain "
        "to the input of the network.  It actually stores the lattice-like "
        "supervision information at the output of the network (which imposes "
        "constraints on which frames each phone can be active on")
        .def(py::init<>())
        .def_readwrite("name", &PyClass::name,
                       "the name of the output in the neural net; in simple "
                       "setups it will just be 'output'.")
        .def_readwrite(
            "indexes", &PyClass::indexes,
            "The indexes that the output corresponds to.  The size of this "
            "vector will be equal to supervision.num_sequences * "
            "supervision.frames_per_sequence. Be careful about the order of "
            "these indexes-- it is a little confusing. The indexes in the "
            "'index' vector are ordered as: (frame 0 of each sequence); (frame "
            "1 of each sequence); and so on.  But in the 'supervision' object, "
            "the FST contains (sequence 0; sequence 1; ...).  So reordering is "
            "needed when doing the numerator computation. We order 'indexes' "
            "in this way for efficiency in the denominator computation (it "
            "helps memory locality), as well as to avoid the need for the nnet "
            "to reorder things internally to match the requested output (for "
            "layers inside the neural net, the ordering is (frame 0; frame 1 "
            "...) as this corresponds to the order you get when you sort a "
            "vector of Index).")
        .def_readwrite("supervision", &PyClass::supervision,
                       "The supervision object, containing the FST.")
        .def_readwrite(
            "deriv_weights", &PyClass::deriv_weights,
            "This is a vector of per-frame weights, required to be between 0 "
            "and 1, that is applied to the derivative during training (but not "
            "during model combination, where the derivatives need to agree "
            "with the computed objf values for the optimization code to work). "
            " The reason for this is to more exactly handle edge effects and "
            "to ensure that no frames are 'double-counted'.  The order of this "
            "vector corresponds to the order of the 'indexes' (i.e. all the "
            "first frames, then all the second frames, etc.) If this vector is "
            "empty it means we're not applying per-frame weights, so it's "
            "equivalent to a vector of all ones.  This vector is written to "
            "disk compactly as unsigned char.")
        .def("CheckDim", &PyClass::CheckDim)
        .def("__str__",
             [](const PyClass& sup) {
               std::ostringstream os;
               os << "name: " << sup.name << "\n";
               return os.str();
             })
        // TODO(fangjun): other methods can be wrapped when needed
        ;
  }
  {
    using PyClass = NnetChainExample;
    py::class_<PyClass>(m, "NnetChainExample")
        .def(py::init<>())
        .def_readwrite("inputs", &PyClass::inputs)
        .def_readwrite("outputs", &PyClass::outputs)
        .def("Compress", &PyClass::Compress,
             "Compresses the input features (if not compressed)")
        .def("__eq__",
             [](const PyClass& a, const PyClass& b) { return a == b; });

    // (fangjun): we follow the PyKaldi style to prepend a underline before the
    // registered classes and the user in general should not use them directly;
    // instead, they should use the corresponding python classes that are more
    // easier to use.
    pybind_sequential_table_reader<KaldiObjectHolder<PyClass>>(
        m, "_SequentialNnetChainExampleReader");

    pybind_random_access_table_reader<KaldiObjectHolder<PyClass>>(
        m, "_RandomAccessNnetChainExampleReader");

    pybind_table_writer<KaldiObjectHolder<PyClass>>(m,
                                                    "_NnetChainExampleWriter");
  }
}
