// pybind/decoder/decoder_wrappers_pybind.cc

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

#include "decoder/decoder_wrappers_pybind.h"

#include "decoder/decoder-wrappers.h"

using namespace kaldi;

namespace {

template <typename FST>
void pybind_decode_utterance_lattice_faster_impl(
    py::module& m, const std::string& func_name,
    const std::string& func_help_doc = "") {
  m.def(
      func_name.c_str(),
      [](LatticeFasterDecoderTpl<FST>& decoder, DecodableInterface& decodable,
         const TransitionModel& trans_model, const fst::SymbolTable* word_syms,
         std::string utt, double acoustic_scale, bool determinize,
         bool allow_partial, Int32VectorWriter* alignments_writer,
         Int32VectorWriter* words_writer,
         CompactLatticeWriter* compact_lattice_writer,
         LatticeWriter* lattice_writer) -> std::pair<bool, double> {
        // return a pair [is_succeeded, likelihood]
        double likelihood = 0;
        bool is_succeeded = DecodeUtteranceLatticeFaster(
            decoder, decodable, trans_model, word_syms, utt, acoustic_scale,
            determinize, allow_partial, alignments_writer, words_writer,
            compact_lattice_writer, lattice_writer, &likelihood);
        return std::make_pair(is_succeeded, likelihood);
      },
      func_help_doc.c_str(), py::arg("decoder"), py::arg("decodable"),
      py::arg("trans_model"), py::arg("word_syms"), py::arg("utt"),
      py::arg("acoustic_scale"), py::arg("determinize"),
      py::arg("allow_partial"), py::arg("alignments_writer"),
      py::arg("words_writer"), py::arg("compact_lattice_writer"),
      py::arg("lattice_writer"));
}

}  // namespace

void pybind_decoder_wrappers(py::module& m) {
  pybind_decode_utterance_lattice_faster_impl<fst::StdFst>(
      m, "DecodeUtteranceLatticeFaster",
      "Return a pair [is_succeeded, likelihood], where is_succeeded is true if "
      "it decoded successfully."
      "\n"
      "This function DecodeUtteranceLatticeFaster is used in several decoders, "
      "and we have moved it here.  Note: this is really 'binary-level' code as "
      "it involves table readers and writers; we've just put it here as there "
      "is no other obvious place to put it.  If determinize == false, it "
      "writes to lattice_writer, else to compact_lattice_writer. The writers "
      "for alignments and words will only be written to if they are open.");
  // TODO(fangjun): add wrapper for fst::GrammarFst
  // Add wrappers for other functions when needed
}
