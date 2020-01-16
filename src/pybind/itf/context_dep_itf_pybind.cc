// pybind/itf/context_dep_itf_pybind.cc

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

#include "itf/context_dep_itf_pybind.h"

#include "itf/context-dep-itf.h"

using namespace kaldi;

void pybind_context_dep_itf(py::module& m) {
  {
    using PyClass = ContextDependencyInterface;
    py::class_<PyClass>(m, "ContextDependencyInterface",
                        "context-dep-itf.h provides a link between the "
                        "tree-building code in ../tree/, and the FST code in "
                        "../fstext/ (particularly, ../fstext/context-dep.h).  "
                        "It is an abstract interface that describes an object "
                        "that can map from a phone-in-context to a sequence of "
                        "integer leaf-ids.")
        .def("ContextWidth", &PyClass::ContextWidth,
             "ContextWidth() returns the value N (e.g. 3 for triphone models) "
             "that says how many phones are considered for computing context.")
        .def("CentralPosition", &PyClass::CentralPosition,
             "Central position P of the phone context, in 0-based numbering, "
             "e.g. P = 1 for typical triphone system.  We have to see if we "
             "can do without this function.")
        .def("Compute",
             [](const PyClass& ctx, const std::vector<int32>& phoneseq,
                int32 pdf_class) -> std::vector<int> {
               std::vector<int> res(2, 0);
               res[0] = ctx.Compute(phoneseq, pdf_class, &res[1]);
               return res;
             },
             "Return a pair [is_succeeded, pdf_id], where is_succeeded is 0 "
             "if expansion somehow failed."
             "\n"
             "The 'new' Compute interface.  For typical topologies, pdf_class "
             "would be 0, 1, 2."
             "\n"
             "'Compute' is the main function of this interface, that takes a "
             "sequence of N phones (and it must be N phones), possibly "
             "including epsilons (symbol id zero) but only at positions other "
             "than P [these represent unknown phone context due to end or "
             "begin of sequence].  We do not insist that Compute must always "
             "output (into stateseq) a nonempty sequence of states, but we "
             "anticipate that stateseq will always be nonempty at output "
             "intypical use cases.  'Compute' returns false if expansion "
             "somehow failed.  Normally the calling code should raise an "
             "exception if this happens.  We can define a different interface "
             "later in order to handle other kinds of information-- the "
             "underlying data-structures from event-map.h are very flexible.",
             py::arg("phoneseq"), py::arg("pdf_class"))
        .def("GetPdfInfo",
             [](const PyClass* ctx,
                const std::vector<int32>& phones,          // list of phones
                const std::vector<int32>& num_pdf_classes  // indexed by phone,
                ) {
               std::vector<std::vector<std::pair<int, int>>> pdf_info;
               ctx->GetPdfInfo(phones, num_pdf_classes, &pdf_info);
               return pdf_info;
             },
             "GetPdfInfo returns a vector indexed by pdf-id, saying for each "
             "pdf which pairs of (phone, pdf-class) it can correspond to.  "
             "(Usually just one). c.f. hmm/hmm-topology.h for meaning of "
             "pdf-class. This is the old, simpler interface of GetPdfInfo(), "
             "and that this one can only be called if the HmmTopology object's "
             "IsHmm() function call returns true.",
             py::arg("phones"), py::arg("num_pdf_classes"))
        .def("GetPdfInfo",
             [](const PyClass* ctx, const std::vector<int32>& phones,
                const std::vector<std::vector<std::pair<int32, int32>>>&
                    pdf_class_pairs) {
               std::vector<std::vector<std::vector<std::pair<int, int>>>>
                   pdf_info;
               ctx->GetPdfInfo(phones, pdf_class_pairs, &pdf_info);
               return pdf_info;
             },
             "This function outputs information about what possible pdf-ids "
             "can be generated for HMM-states; it covers the general case "
             "where the self-loop pdf-class may be different from the "
             "forward-transition pdf-class, so we are asking not about the set "
             "of possible pdf-ids for a given (phone, pdf-class), but the set "
             "of possible ordered pairs (forward-transition-pdf, "
             "self-loop-pdf) for a given (phone, forward-transition-pdf-class, "
             "self-loop-pdf-class). Note: 'phones' is a list of integer ids of "
             "phones, and 'pdf-class-pairs', indexed by phone, is a list of "
             "pairs (forward-transition-pdf-class, self-loop-pdf-class) that "
             "we can have for that phone. The output 'pdf_info' is indexed "
             "first by phone and then by the same index that indexes each "
             "element of 'pdf_class_pairs', and tells us for each pair in "
             "'pdf_class_pairs', what is the list of possible "
             "(forward-transition-pdf-id, self-loop-pdf-id) that we can have. "
             "This is less efficient than the other version of GetPdfInfo().",
             py::arg("phones"), py::arg("pdf_class_pairs"))
        .def("NumPdfs", &PyClass::NumPdfs,
             "NumPdfs() returns the number of acoustic pdfs (they are numbered "
             "0.. NumPdfs()-1).")
        .def("Copy", &PyClass::Copy,
             "Returns pointer to new object which is copy of current one.",
             py::return_value_policy::take_ownership);
  }
}
