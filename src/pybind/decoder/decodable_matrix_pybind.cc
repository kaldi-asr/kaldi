// pybind/decoder/decodable_matrix_pybind.cc

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

#include "decoder/decodable_matrix_pybind.h"

#include "decoder/decodable-matrix.h"

using namespace kaldi;

namespace {

void pybind_decodable_matrix_scale_mapped(py::module& m) {
  using PyClass = DecodableMatrixScaledMapped;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixScaledMapped")
      .def(py::init<const TransitionModel&, const Matrix<BaseFloat>&,
                    BaseFloat>(),
           "This constructor creates an object that will not delete 'likes' "
           "when done.",
           py::arg("tm"), py::arg("likes"), py::arg("scale"))
      // TODO(fangjun): how to wrap the constructor taking the ownership of
      // likes??
      ;
}

void pybind_decodable_matrix_mapped(py::module& m) {
  using PyClass = DecodableMatrixMapped;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixMapped",
                                          R"doc(
   This is like DecodableMatrixScaledMapped, but it doesn't support an acoustic
   scale, and it does support a frame offset, whereby you can state that the
   first row of 'likes' is actually the n'th row of the matrix of available
   log-likelihoods.  It's useful if the neural net output comes in chunks for
   different frame ranges.

   Note: DecodableMatrixMappedOffset solves the same problem in a slightly
   different way, where you use the same decodable object.  This one, unlike
   DecodableMatrixMappedOffset, is compatible with when the loglikes are in a
   SubMatrix.
      )doc")
      .def(py::init<const TransitionModel&, const Matrix<BaseFloat>&, int>(),
           R"doc(
  This constructor creates an object that will not delete "likes" when done.
  the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  greater than one if this is not the first chunk of likelihoods.
                    )doc",
           py::arg("tm"), py::arg("likes"), py::arg("frame_offset") = 0)
      // TODO(fangjun): how to wrap the constructor taking the ownership of
      // the likes??
      ;
}

void pybind_decodable_matrix_mapped_offset(py::module& m) {
  using PyClass = DecodableMatrixMappedOffset;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixMappedOffset",
                                          R"doc(
   This decodable class returns log-likes stored in a matrix; it supports
   repeatedly writing to the matrix and setting a time-offset representing the
   frame-index of the first row of the matrix.  It's intended for use in
   multi-threaded decoding; mutex and semaphores are not included.  External
   code will call SetLoglikes() each time more log-likelihods are available.
   If you try to access a log-likelihood that's no longer available because
   the frame index is less than the current offset, it is of course an error.

   See also DecodableMatrixMapped, which supports the same type of thing but
   with a different interface where you are expected to re-construct the
   object each time you want to decode.
      )doc")
      .def(py::init<const TransitionModel&>(), py::arg("tm"))
      .def("FirstAvailableFrame", &PyClass::FirstAvailableFrame,
           "this is not part of the generic Decodable interface.")
      .def("AcceptLoglikes", &PyClass::AcceptLoglikes,
           R"doc(
  Logically, this function appends 'loglikes' (interpreted as newly available
  frames) to the log-likelihoods stored in the class.

  This function is destructive of the input "loglikes" because it may
  under some circumstances do a shallow copy using Swap().  This function
  appends loglikes to any existing likelihoods you've previously supplied.
          )doc",
           py::arg("loglikes"), py::arg("frames_to_discard"))
      .def("InputIsFinished", &PyClass::InputIsFinished);
}

void pybind_decodable_matrix_scaled(py::module& m) {
  using PyClass = DecodableMatrixScaled;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixScaled")
      .def(py::init<const Matrix<BaseFloat>&, BaseFloat>(), py::arg("likes"),
           py::arg("scale"));
}

}  // namespace

void pybind_decodable_matrix(py::module& m) {
  pybind_decodable_matrix_scale_mapped(m);
  pybind_decodable_matrix_mapped(m);
  pybind_decodable_matrix_mapped_offset(m);
  pybind_decodable_matrix_scaled(m);
}
