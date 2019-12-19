// pybind/feat/wave_reader_pybind.cc

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

#include "feat/wave_reader_pybind.h"
#include "feat/wave-reader.h"
#include "util/table-types.h"

using namespace kaldi;

void pybind_wave_reader(py::module& m) {
  m.attr("kWaveSampleMax") = py::cast(kWaveSampleMax);

  py::class_<WaveInfo>(m, "WaveInfo")
      .def(py::init<>())
      .def("IsStreamed", &WaveInfo::IsStreamed,
           "Is stream size unknown? Duration and SampleCount not valid if true.")
      .def("SampFreq", &WaveInfo::SampFreq,
           "Sample frequency, Hz.")
      .def("SampleCount", &WaveInfo::SampleCount,
           "Number of samples in stream. Invalid if IsStreamed() is true.")
      .def("Duration", &WaveInfo::Duration,
           "Approximate duration, seconds. Invalid if IsStreamed() is true.")
      .def("NumChannels", &WaveInfo::NumChannels,
           "Number of channels, 1 to 16.")
      .def("BlockAlign", &WaveInfo::BlockAlign,
           "Bytes per sample.")
      .def("DataBytes", &WaveInfo::DataBytes,
           "Wave data bytes. Invalid if IsStreamed() is true.")
      .def("ReverseBytes", &WaveInfo::ReverseBytes,
           "Is data file byte order different from machine byte order?");

  py::class_<WaveData>(m, "WaveData")
      .def(py::init<>())
      .def(py::init<const float, const Matrix<float>>(),
           py::arg("samp_freq"), py::arg("data"))
      .def("Duration", &WaveData::Duration,
           "Returns the duration in seconds")
      .def("Data", &WaveData::Data, py::return_value_policy::reference)
      .def("SampFreq", &WaveData::SampFreq)
      .def("Clear", &WaveData::Clear)
      .def("CopyFrom", &WaveData::CopyFrom)
      .def("Swap", &WaveData::Swap);

  py::class_<SequentialTableReader<WaveHolder>>(m, "SequentialWaveReader")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("rspecifier"))
      .def("Open", &SequentialTableReader<WaveHolder>::Open, py::arg("rspecifier"))
      .def("Done", &SequentialTableReader<WaveHolder>::Done)
      .def("Key", &SequentialTableReader<WaveHolder>::Key)
      .def("FreeCurrent", &SequentialTableReader<WaveHolder>::FreeCurrent)
      .def("Value", &SequentialTableReader<WaveHolder>::Value,
           py::return_value_policy::reference)
      .def("Next", &SequentialTableReader<WaveHolder>::Next)
      .def("IsOpen", &SequentialTableReader<WaveHolder>::IsOpen)
      .def("Close", &SequentialTableReader<WaveHolder>::Close);

  py::class_<RandomAccessTableReader<WaveHolder>>(m, "RandomAccessWaveReader")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("rspecifier"))
      .def("Open", &RandomAccessTableReader<WaveHolder>::Open, py::arg("rspecifier"))
      .def("IsOpen", &RandomAccessTableReader<WaveHolder>::IsOpen)
      .def("Close", &RandomAccessTableReader<WaveHolder>::Close)
      .def("HasKey", &RandomAccessTableReader<WaveHolder>::HasKey, py::arg("key"))
      .def("Value", &RandomAccessTableReader<WaveHolder>::Value,
           py::return_value_policy::reference);

  py::class_<SequentialTableReader<WaveInfoHolder>>(m, "SequentialWaveInfoReader")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("rspecifier"))
      .def("Open", &SequentialTableReader<WaveInfoHolder>::Open, py::arg("rspecifier"))
      .def("Done", &SequentialTableReader<WaveInfoHolder>::Done)
      .def("Key", &SequentialTableReader<WaveInfoHolder>::Key)
      .def("FreeCurrent", &SequentialTableReader<WaveInfoHolder>::FreeCurrent)
      .def("Value", &SequentialTableReader<WaveInfoHolder>::Value,
           py::return_value_policy::reference)
      .def("Next", &SequentialTableReader<WaveInfoHolder>::Next)
      .def("IsOpen", &SequentialTableReader<WaveInfoHolder>::IsOpen)
      .def("Close", &SequentialTableReader<WaveInfoHolder>::Close);

  py::class_<RandomAccessTableReader<WaveInfoHolder>>(m, "RandomAccessWaveInfoReader")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("rspecifier"))
      .def("Open", &RandomAccessTableReader<WaveInfoHolder>::Open, py::arg("rspecifier"))
      .def("IsOpen", &RandomAccessTableReader<WaveInfoHolder>::IsOpen)
      .def("Close", &RandomAccessTableReader<WaveInfoHolder>::Close)
      .def("HasKey", &RandomAccessTableReader<WaveInfoHolder>::HasKey, py::arg("key"))
      .def("Value", &RandomAccessTableReader<WaveInfoHolder>::Value,
           py::return_value_policy::reference);

}

