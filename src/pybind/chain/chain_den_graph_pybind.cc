// pybind/chain/chain_den_graph_pybind.cc

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

#include "chain/chain-den-graph.h"
#include "fst/vector_fst_pybind.h"

using namespace fst;
using namespace kaldi::chain;

void pybind_chain_den_graph(py::module& m) {
  using PyClass = DenominatorGraph;
  py::class_<PyClass>(m, "DenominatorGraph",
                      "This class is responsible for storing the FST that we use as the"
                      "'anti-model' or 'denominator-model', that models all possible phone"
                      "sequences (or most possible phone sequences, depending how we built it).."
                      "It stores the FST in a format where we can access both the transitions out"
                      "of each state, and the transitions into each state.")
    .def(py::init<const StdVectorFst&, int>(),
         "Initialize from epsilon-free acceptor FST with pdf-ids plus one as the"
         "labels.  'num_pdfs' is only needeed for checking.",
         py::arg("fst"), py::arg("num_pdfs"));
}
