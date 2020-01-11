// pybind/hmm/hmm_topology_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "hmm/hmm_topology_pybind.h"

#include "hmm/hmm-topology.h"

using namespace kaldi;

void pybind_hmm_topology(py::module& m) {
  using PyClass = HmmTopology;
  auto hmm = py::class_<PyClass>(
      m, "HmmTopology",
      "A class for storing topology information for phones. See `hmm` for "
      "context. This object is sometimes accessed in a file by itself, but "
      "more often as a class member of the Transition class (this is for "
      "convenience to reduce the number of files programs have to access).");

  using State = HmmTopology::HmmState;
  py::class_<State>(
      hmm, "HmmState",
      "A structure defined inside HmmTopology to represent a HMM state.")
      .def(py::init<>())
      .def(py::init<int>(), py::arg("pdf_class"))
      .def(py::init<int, int>(), py::arg("forward_pdf_class"),
           py::arg("self_loop_pdf_class"))
      .def_readwrite("forward_pdf_class", &State::forward_pdf_class)
      .def_readwrite("self_loop_pdf_class", &State::self_loop_pdf_class)
      .def_readwrite("transitions", &State::transitions)
      .def("__eq__", [](const State& s1, const State& s2) { return s1 == s2; })
      .def("__str__", [](const State& s) {
        std::ostringstream os;
        os << "forward_pdf_class: " << s.forward_pdf_class << "\n";
        os << "self_loop_pdf_class: " << s.self_loop_pdf_class << "\n";
        return os.str();
      });

  hmm.def(py::init<>())
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"))
      .def("Check", &PyClass::Check,
           "Checks that the object is valid, and throw exception otherwise.")
      .def("IsHmm", &PyClass::IsHmm,
           "Returns true if this HmmTopology is really 'hmm-like', i.e. the "
           "pdf-class on the self-loops and forward transitions of all states "
           "are identical. [note: in HMMs, the densities are associated with "
           "the states.] We have extended this to support 'non-hmm-like' "
           "topologies (where those pdf-classes are different), in order to "
           "make for more compact decoding graphs in our so-called 'chain "
           "models' (AKA lattice-free MMI), where we use 1-state topologies "
           "that have different pdf-classes for the self-loop and the forward "
           "transition. Note that we always use the 'reorder=true' option so "
           "the 'forward transition' actually comes before the self-loop.")
      .def("TopologyForPhone", &PyClass::TopologyForPhone,
           "Returns the topology entry (i.e. vector of HmmState) for this "
           "phone; will throw exception if phone not covered by the topology.",
           py::arg("phone"), py::return_value_policy::reference)
      .def("NumPdfClasses", &PyClass::NumPdfClasses,
           "Returns the number of 'pdf-classes' for this phone; throws "
           "exception if phone not covered by this topology.",
           py::arg("phone"))
      .def("GetPhones", &PyClass::GetPhones,
           "Returns a reference to a sorted, unique list of phones covered by "
           "the topology (these phones will be positive integers, and usually "
           "contiguous and starting from one but the toolkit doesn't assume "
           "they are contiguous).",
           py::return_value_policy::reference)
      .def("GetPhoneToNumPdfClasses",
           [](const PyClass& topo) -> std::vector<int> {
             std::vector<int> phone2num_pdf_classes;
             topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
             return phone2num_pdf_classes;
           },
           "Outputs a vector of int32, indexed by phone, that gives the number "
           "of \ref pdf_class pdf-classes for the phones; this is used by "
           "tree-building code such as BuildTree().")
      .def("MinLength", &PyClass::MinLength,
           "Returns the minimum number of frames it takes to traverse this "
           "model for this phone: e.g. 3 for the normal HMM topology.",
           py::arg("phone"))
      .def("__eq__",
           [](const PyClass& t1, const PyClass& t2) { return t1 == t2; })
      .def("__str__", [](const PyClass& topo) {
        std::ostringstream os;
        bool binary = false;
        topo.Write(os, binary);
        return os.str();
      });
}
