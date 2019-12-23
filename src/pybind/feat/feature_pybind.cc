// pybind/feat/feature_pybind.cc

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

#include "feat/feature_pybind.h"

#include "feat/feature-mfcc.h"
#include "feat/feature-fbank.h"

using namespace kaldi;

template <class Feature>
void offline_feature(py::module& m, const std::string& feat_type) {
  py::class_<OfflineFeatureTpl<Feature>>(m, feat_type.c_str())
      .def(py::init<const typename Feature::Options&>())
      .def("ComputeFeatures", &OfflineFeatureTpl<Feature>::ComputeFeatures)
      .def("Dim", &OfflineFeatureTpl<Feature>::Dim);
}

void pybind_feature(py::module& m) {
  py::class_<FrameExtractionOptions>(m, "FrameExtractionOptions")
      .def_readwrite("samp_freq", &FrameExtractionOptions::samp_freq)
      .def_readwrite("frame_shift_ms", &FrameExtractionOptions::frame_shift_ms)
      .def_readwrite("frame_length_ms", &FrameExtractionOptions::frame_length_ms)
      .def_readwrite("dither", &FrameExtractionOptions::dither)
      .def_readwrite("preemph_coeff", &FrameExtractionOptions::preemph_coeff)
      .def_readwrite("remove_dc_offset", &FrameExtractionOptions::remove_dc_offset)
      .def_readwrite("window_type", &FrameExtractionOptions::window_type)
      .def_readwrite("round_to_power_of_two", &FrameExtractionOptions::round_to_power_of_two)
      .def_readwrite("blackman_coeff", &FrameExtractionOptions::blackman_coeff)
      .def_readwrite("snip_edges", &FrameExtractionOptions::snip_edges)
      .def_readwrite("allow_downsample", &FrameExtractionOptions::allow_downsample)
      .def_readwrite("allow_upsample", &FrameExtractionOptions::allow_upsample)
      .def_readwrite("max_feature_vectors", &FrameExtractionOptions::max_feature_vectors);

  py::class_<MelBanksOptions>(m, "MelBanksOptions")
      .def(py::init<const int&>())
      .def_readwrite("num_bins", &MelBanksOptions::num_bins)
      .def_readwrite("low_freq", &MelBanksOptions::low_freq)
      .def_readwrite("high_freq", &MelBanksOptions::high_freq)
      .def_readwrite("vtln_low", &MelBanksOptions::vtln_low)
      .def_readwrite("vtln_high", &MelBanksOptions::vtln_high)
      .def_readwrite("debug_mel", &MelBanksOptions::debug_mel)
      .def_readwrite("htk_mode", &MelBanksOptions::htk_mode);

  py::class_<MfccOptions>(m, "MfccOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &MfccOptions::frame_opts)
      .def_readwrite("mel_opts", &MfccOptions::mel_opts)
      .def_readwrite("num_ceps", &MfccOptions::num_ceps)
      .def_readwrite("use_energy", &MfccOptions::use_energy)
      .def_readwrite("energy_floor", &MfccOptions::energy_floor)
      .def_readwrite("raw_energy", &MfccOptions::raw_energy)
      .def_readwrite("cepstral_lifter", &MfccOptions::cepstral_lifter)
      .def_readwrite("htk_compat", &MfccOptions::htk_compat);

  py::class_<FbankOptions>(m, "FbankOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &FbankOptions::frame_opts)
      .def_readwrite("mel_opts", &FbankOptions::mel_opts)
      .def_readwrite("use_energy", &FbankOptions::use_energy)
      .def_readwrite("energy_floor", &FbankOptions::energy_floor)
      .def_readwrite("raw_energy", &FbankOptions::raw_energy)
      .def_readwrite("use_log_fbank", &FbankOptions::use_log_fbank)
      .def_readwrite("use_power", &FbankOptions::use_power)
      .def_readwrite("htk_compat", &FbankOptions::htk_compat);

  offline_feature<MfccComputer>(m, "Mfcc");
  offline_feature<FbankComputer>(m, "Fbank");
}
