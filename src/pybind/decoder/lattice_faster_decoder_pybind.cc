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
      .def("__str__", [](const PyClass& opts) {
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
        os << "  word_determinize: " << opts.det_opts.word_determinize << "\n";
        os << "  minimize: " << opts.det_opts.minimize << "\n";
        return os.str();
      });
}

}  // namespace

void pybind_lattice_faster_decoder(py::module& kaldi_module,
                                   py::module& decoder_module) {
  pybind_lattice_faster_decoder_config(kaldi_module);
}
