// pybind/decoder/lattice_faster_decoder_pybind.h

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

#include "decoder/lattice_faster_decoder_pybind.h"

#include "decoder/lattice-faster-decoder.h"
#include "fst/fst.h"

using namespace kaldi;

namespace {

void pybind_lattice_faster_decoder_config(py::module& m) {
  using PyClass = LatticeFasterDecoderConfig;

  py::class_<PyClass>(m, "LatticeFasterDecoderConfig")
      .def(py::init<>())
      .def_readwrite("beam", &PyClass::beam,
                     "Decoding beam.  Larger->slower, more accurate.")
      .def_readwrite("max_active", &PyClass::max_active,
                     "Decoder max active states. Larger->slower; more accurate")
      .def_readwrite("min_active", &PyClass::min_active,
                     "Decoder minimum #active states.")
      .def_readwrite(
          "lattice_beam", &PyClass::lattice_beam,
          "Lattice generation beam.  Larger->slower, and deeper lattices",
          "Interval (in frames) at which to prune tokens")
      .def_readwrite("prune_interval", &PyClass::prune_interval,
                     "Interval (in frames) at which to prune tokens")
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice,
                     "If true, determinize the lattice "
                     "(lattice-determinization, keeping only best pdf-sequence "
                     "for each word-sequence).")
      .def_readwrite("beam_delta", &PyClass::beam_delta,
                     "Increment used in decoding-- this parameter is obscure "
                     "and relates to a speedup in the way the max-active "
                     "constraint is applied.  Larger is more accurate.")
      .def_readwrite("hash_ratio", &PyClass::hash_ratio,
                     "Setting used in decoder to control hash behavior")
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def("Check", &PyClass::Check)
      .def("__str__",
           [](const PyClass& opts) {
             std::ostringstream os;
             os << "beam: " << opts.beam << "\n";
             os << "max_active: " << opts.max_active << "\n";
             os << "lattice_beam: " << opts.lattice_beam << "\n";
             os << "prune_interval: " << opts.prune_interval << "\n";
             os << "determinize_lattice: " << opts.determinize_lattice << "\n";
             os << "beam_delta: " << opts.beam_delta << "\n";
             os << "hash_ratio: " << opts.hash_ratio << "\n";
             os << "prune_scale: " << opts.prune_scale << "\n";

             os << "det_opts:\n";
             os << "  delta: " << opts.det_opts.delta << "\n";
             os << "  max_mem: " << opts.det_opts.max_mem << "\n";
             os << "  phone_determinize: " << opts.det_opts.phone_determinize
                << "\n";
             os << "  word_determinize: " << opts.det_opts.word_determinize
                << "\n";
             os << "  minimize: " << opts.det_opts.minimize << "\n";
             return os.str();
           })
      .def("Register", &PyClass::Register, py::arg("opts"));
}

template <typename FST, typename Token = decoder::StdToken>
void pybind_lattice_faster_decoder_impl(
    py::module& m, const std::string& class_name,
    const std::string& class_help_doc = "") {
  using PyClass = LatticeFasterDecoderTpl<FST, Token>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&, const LatticeFasterDecoderConfig&>(),
           "Instantiate this class once for each thing you have to decode. "
           "This version of the constructor does not take ownership of 'fst'.",
           py::arg("fst"), py::arg("config"))
      // TODO(fangjun): how to wrap the constructor taking the ownership of fst
      .def("SetOptions", &PyClass::SetOptions, py::arg("config"))
      .def("GetOptions", &PyClass::GetOptions,
           py::return_value_policy::reference)
      .def("Decode", &PyClass::Decode,
           "Decodes until there are no more frames left in the 'decodable' "
           "object..\n"
           "note, this may block waiting for input if the 'decodable' object "
           "blocks. Returns true if any kind of traceback is available (not "
           "necessarily from a final state).",
           py::arg("decodable"))
      .def("ReachedFinal", &PyClass::ReachedFinal,
           "says whether a final-state was active on the last frame.  If it "
           "was not, the lattice (or traceback) will end with states that are "
           "not final-states.")
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
           "Return a pair [is_succeeded, lattice], where is_succeeded is true "
           "if lattice is NOT empty.\n"
           "If the lattice is not empty, it contains the single best path "
           "through the lattice."
           "\n"
           "If 'use_final_probs' is true AND we reached the final-state of the "
           "graph then it will include those as final-probs, else it will "
           "treat all final-probs as one.  Note: this just calls "
           "GetRawLattice() and figures out the shortest path."
           "\n"
           "Note: Using the return status `is_succeeded` is deprecated, it "
           "will be removed",
           py::arg("use_final_probs") = true)
      .def("GetRawLattice",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetRawLattice(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
           R"doc(
  Return a pair [is_succeeded, lattice], where is_succeeded is true
  if lattice is not empty.

  Outputs an FST corresponding to the raw, state-level
  tracebacks.  Returns true if result is nonempty.
  If "use_final_probs" is true AND we reached the final-state
  of the graph then it will include those as final-probs, else
  it will treat all final-probs as one.
  The raw lattice will be topologically sorted.

  See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  which also supports a pruning beam, in case for some reason
  you want it pruned tighter than the regular lattice beam.
  We could put that here in future needed.
           )doc",
           py::arg("use_final_probs") = true)
      // NOTE(fangjun): we do not wrap the deprecated `GetLattice` method
      .def("InitDecoding", &PyClass::InitDecoding,
           "InitDecoding initializes the decoding, and should only be used if "
           "you intend to call AdvanceDecoding().  If you call Decode(), you "
           "don't need to call this.  You can also call InitDecoding if you "
           "have already decoded an utterance and want to start with a new "
           "utterance.")
      .def("AdvanceDecoding", &PyClass::AdvanceDecoding,
           "This will decode until there are no more frames ready in the "
           "decodable object.  You can keep calling it each time more frames "
           "become available. If max_num_frames is specified, it specifies the "
           "maximum number of frames the function will decode before "
           "returning.",
           py::arg("decodable"), py::arg("max_num_frames") = -1)
      .def("FinalizeDecoding", &PyClass::FinalizeDecoding,
           "This function may be optionally called after AdvanceDecoding(), "
           "when you do not plan to decode any further.  It does an extra "
           "pruning step that will help to prune the lattices output by "
           "GetLattice and (particularly) GetRawLattice more completely, "
           "particularly toward the end of the utterance.  If you call this, "
           "you cannot call AdvanceDecoding again (it will fail), and you "
           "cannot call GetLattice() and related functions with "
           "use_final_probs = false.  Used to be called "
           "PruneActiveTokensFinal().")
      .def("FinalRelativeCost", &PyClass::FinalRelativeCost,
           "FinalRelativeCost() serves the same purpose as ReachedFinal(), but "
           "gives more information.  It returns the difference between the "
           "best (final-cost plus cost) of any token on the final frame, and "
           "the best cost of any token on the final frame.  If it is infinity "
           "it means no final-states were present on the final frame.  It will "
           "usually be nonnegative.  If it not too positive (e.g. < 5 is my "
           "first guess, but this is not tested) you can take it as a good "
           "indication that we reached the final-state with reasonable "
           "likelihood.")
      .def("NumFramesDecoded", &PyClass::NumFramesDecoded,
           "Returns the number of frames decoded so far.  The value returned "
           "changes whenever we call ProcessEmitting().");
}

}  // namespace

void pybind_lattice_faster_decoder(py::module& m) {
  pybind_lattice_faster_decoder_config(m);

  using namespace decoder;
  {
    py::module decoder_m = m.def_submodule("decoder", "pybind for decoder");
    // You are not supposed to use the following classes directly in Python
    auto std_token = py::class_<StdToken>(decoder_m, "_StdToken");
    auto backpointer_token =
        py::class_<BackpointerToken>(decoder_m, "_BackpointerToken");
    auto forward_link_std_token =
        py::class_<ForwardLink<StdToken>>(decoder_m, "_ForwardLinkStdToken");
    auto forward_link_backpointer_token =
        py::class_<ForwardLink<BackpointerToken>>(
            decoder_m, "_ForwardLinkBackpointerToken");
  }

  pybind_lattice_faster_decoder_impl<fst::StdFst, StdToken>(
      m, "LatticeFasterDecoder",
      R"doc(This is the "normal" lattice-generating decoder.
See `lattices_generation` `decoders_faster` and `decoders_simple`
for more information.

The decoder is templated on the FST type and the token type.  The token type
will normally be StdToken, but also may be BackpointerToken which is to support
quick lookup of the current best path (see lattice-faster-online-decoder.h)

The FST you invoke this decoder which is expected to equal
Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
FST == StdFst and it notices that the actual FST type is
fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
will internally cast itself to one that is templated on those more specific
types; this is an optimization for speed.)doc");
}
