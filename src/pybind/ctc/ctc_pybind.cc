// pybind/ctc/ctc_pybind.cc

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

#include "ctc/ctc_pybind.h"

#include "ctc/warp-ctc/include/ctc.h"
#include "dlpack/dlpack_subvector.h"

using namespace kaldi;

namespace {

// note that the argument order does not follow the google style
// which puts the output argument last;
ctcStatus_t compute_ctc_loss_cpu(const DLPackSubVector<float>& activations,
                                 DLPackSubVector<float>* gradients,
                                 const DLPackSubVector<int>& flat_labels,
                                 const DLPackSubVector<int>& label_lengths,
                                 const DLPackSubVector<int>& input_lengths,
                                 int alphabet_size, int minibatch,
                                 DLPackSubVector<float>* costs,
                                 DLPackSubVector<float>* workspace,
                                 ctcOptions options) {
  ctcStatus_t status = compute_ctc_loss(
      activations.Data(), gradients->Data(), flat_labels.Data(),
      label_lengths.Data(), input_lengths.Data(), alphabet_size, minibatch,
      costs->Data(), workspace->Data(), options);
  return status;
}

ctcStatus_t compute_ctc_loss_gpu(const DLPackCuSubVector<float>& activations,
                                 DLPackCuSubVector<float>* gradients,
                                 const DLPackSubVector<int>& flat_labels,
                                 const DLPackSubVector<int>& label_lengths,
                                 const DLPackSubVector<int>& input_lengths,
                                 int alphabet_size, int minibatch,
                                 DLPackSubVector<float>* costs,
                                 DLPackCuSubVector<float>* workspace,
                                 ctcOptions options) {
  // we use the default CUDA stream by setting it to nullptr, i.e., 0
  //
  // As far as PyTorch is concerned, we can get the underlying CUDA stream in
  // PyTorch (the `_cdata` attribute) and pass it to kaldi pybind. But this
  // approach is too tricky and is specific to PyTorch.
  //
  // FYI (fangjun): the relevant information to implement the above approach is
  // here
  //
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/cuda/Module.cpp#L117
  // https://github.com/pytorch/pytorch/blob/master/torch/cuda/__init__.py#L431
  // https://github.com/pytorch/pytorch/blob/master/c10/core/Stream.h#L104
  // https://github.com/pytorch/pytorch/blob/master/c10/core/Stream.h#L126
  options.stream = nullptr;  // use the default stream
  ctcStatus_t status = compute_ctc_loss(
      activations.Data(), gradients->Data(), flat_labels.Data(),
      label_lengths.Data(), input_lengths.Data(), alphabet_size, minibatch,
      costs->Data(), workspace->Data(), options);
  return status;
}

}  // namespace

void pybind_ctc(py::module& kaldi_m) {
  py::module m = kaldi_m.def_submodule("ctc", "pybind for warp-ctc");

  m.def("GetWarpCtcVersion", &get_warpctc_version,
        "Returns a single integer which specifies the API version of the "
        "warpctc library");

  py::enum_<ctcStatus_t>(m, "CtcStatus", py::arithmetic())
      .value("CTC_STATUS_SUCCESS", CTC_STATUS_SUCCESS)
      .value("CTC_STATUS_MEMOPS_FAILED", CTC_STATUS_MEMOPS_FAILED)
      .value("CTC_STATUS_INVALID_VALUE", CTC_STATUS_INVALID_VALUE)
      .value("CTC_STATUS_EXECUTION_FAILED", CTC_STATUS_EXECUTION_FAILED)
      .value("CTC_STATUS_UNKNOWN_ERROR", CTC_STATUS_UNKNOWN_ERROR);

  m.def(
      "CtcGetStatusString", &ctcGetStatusString,
      "Returns a string containing a description of status that was passed in",
      py::arg("status"));

  py::enum_<ctcComputeLocation>(m, "CtcComputeLocation", py::arithmetic())
      .value("CTC_CPU", CTC_CPU)
      .value("CTC_GPU", CTC_GPU);

  py::class_<ctcOptions>(m, "CtcOptions",
                         "Structure used for options to the CTC computation. "
                         "Applications should zero out the array using memset "
                         "and sizeof(struct ctcOptions) in C or default "
                         "initialization (e.g. 'ctcOptions options{};' or "
                         "'auto options = ctcOptions{}') in C++ to ensure "
                         "forward compatibility with added options.")
      .def(py::init<>())
      .def_readwrite("loc", &ctcOptions::loc,
                     "indicates where the ctc calculation should take place "
                     "{CTC_CPU | CTC_GPU}")
      .def_readwrite("num_threads", &ctcOptions::num_threads,
                     "used when loc == CTC_GPU, which stream the kernels "
                     "should be launched in")
      .def_readwrite("blank_label", &ctcOptions::blank_label,
                     "the label value/index that the CTC calculation should "
                     "use as the blank label");

  m.def(
      "ComputeCtcLossCpu", &compute_ctc_loss_cpu,
      "Compute the connectionist temporal classification loss between a "
      "sequence of probabilities and a ground truth labeling on **CPU**.  "
      "Optionally compute the gradient with respect to the inputs."
      "\n"
      "Args:"
      "\n"
      "activations:  pointer to the activations in **CPU** "
      "addressable memory, depending on info.  We assume a fixed memory "
      "layout for this 3 dimensional tensor, which has dimension (t, n, p), "
      "where t is the time index, n is the minibatch index, and p indexes over "
      "probabilities of each symbol in the alphabet. The memory layout is "
      "(t, n, p) in C order (slowest to fastest changing index, aka "
      "row-major), or (p, n, t) in Fortran order (fastest to slowest changing "
      "index, aka column-major). We also assume strides are equal to "
      "dimensions - there is no padding between dimensions. More precisely, "
      "element (t, n, p), for a problem with mini_batch examples  in the mini "
      "batch, and alphabet_size symbols in the alphabet, is located at: "
      "activations[(t * mini_batch + n) * alphabet_size + p]"
      "\n"
      "gradients: if not NULL, then gradients are computed.  Should be "
      "allocated in the same memory space as activations, i.e., **CPU**, "
      "and memory ordering is identical."
      "\n"
      "flat_labels: Always in **CPU** memory. A concatenation of all the "
      "labels for the minibatch."
      "\n"
      "label_lengths: Always in **CPU** memory. The length of each label for "
      "each example in the minibatch."
      "\n"
      "input_lengths: Always in **CPU** memory. The number of time steps for "
      "each sequence in the minibatch."
      "\n"
      "alphabet_size: The number of possible output symbols. There should "
      "be this many probabilities for each time step."
      "\n"
      "mini_batch: How many examples in a minibatch."
      "\n"
      "costs: Always in **CPU** memory. The cost of each example in the "
      "minibatch."
      "\n"
      "workspace In same memory space as activations, i.e., **CPU**. Should be "
      "of size requested by get_workspace_size."
      "\n"
      "options: see struct ctcOptions"
      "\n"
      "Returns:"
      "\n"
      "Status information",
      py::arg("activations"), py::arg("gradients"), py::arg("flat_labels"),
      py::arg("label_lengths"), py::arg("input_lengths"),
      py::arg("alphabet_size"), py::arg("minibatch"), py::arg("costs"),
      py::arg("workspace"), py::arg("options"));

  m.def(
      "ComputeCtcLossGpu", &compute_ctc_loss_gpu,
      "Compute the connectionist temporal classification loss between a "
      "sequence of probabilities and a ground truth labeling on **GPU**.  "
      "Optionally compute the gradient with respect to the inputs."
      "\n"
      "Args:"
      "\n"
      "activations:  pointer to the activations in **GPU** "
      "addressable memory, depending on info.  We assume a fixed memory "
      "layout for this 3 dimensional tensor, which has dimension (t, n, p), "
      "where t is the time index, n is the minibatch index, and p indexes over "
      "probabilities of each symbol in the alphabet. The memory layout is "
      "(t, n, p) in C order (slowest to fastest changing index, aka "
      "row-major), or (p, n, t) in Fortran order (fastest to slowest changing "
      "index, aka column-major). We also assume strides are equal to "
      "dimensions - there is no padding between dimensions. More precisely, "
      "element (t, n, p), for a problem with mini_batch examples  in the mini "
      "batch, and alphabet_size symbols in the alphabet, is located at: "
      "activations[(t * mini_batch + n) * alphabet_size + p]"
      "\n"
      "gradients: if not NULL, then gradients are computed.  Should be "
      "allocated in the same memory space as activations, i.e., **GPU**, "
      "and memory ordering is identical."
      "\n"
      "flat_labels: Always in **CPU** memory. A concatenation of all the "
      "labels for the minibatch."
      "\n"
      "label_lengths: Always in **CPU** memory. The length of each label for "
      "each example in the minibatch."
      "\n"
      "input_lengths: Always in **CPU** memory. The number of time steps for "
      "each sequence in the minibatch."
      "\n"
      "alphabet_size: The number of possible output symbols. There should "
      "be this many probabilities for each time step."
      "\n"
      "mini_batch: How many examples in a minibatch."
      "\n"
      "costs: Always in **CPU** memory. The cost of each example in the "
      "minibatch."
      "\n"
      "workspace In same memory space as activations, i.e., **GPU**. Should be "
      "of size requested by get_workspace_size."
      "\n"
      "options: see struct ctcOptions"
      "\n"
      "Returns:"
      "\n"
      "Status information",
      py::arg("activations"), py::arg("gradients"), py::arg("flat_labels"),
      py::arg("label_lengths"), py::arg("input_lengths"),
      py::arg("alphabet_size"), py::arg("minibatch"), py::arg("costs"),
      py::arg("workspace"), py::arg("options"));

  m.def("GetWorkspaceSize",
        [](const DLPackSubVector<int>& label_lengths,
           const DLPackSubVector<int>& input_lengths, int alphabet_size,
           int minibatch, ctcOptions info) -> std::pair<ctcStatus_t, size_t> {
          size_t size_bytes;
          ctcStatus_t status =
              get_workspace_size(label_lengths.Data(), input_lengths.Data(),
                                 alphabet_size, minibatch, info, &size_bytes);
          return std::make_pair(status, size_bytes);
        },
        "For a given set of labels and minibatch size return the required "
        "workspace size.  This will need to be allocated in the same memory "
        "space as your probabilities."
        "\n"
        "Args:"
        "\n"
        "label_lengths Always in CPU memory. The length of each label for each "
        "example in the minibatch."
        "\n"
        "input_lengths Always in CPU memory.  The number of time steps for "
        "each sequence in the minibatch."
        "\n"
        "alphabet_size How many symbols in the alphabet or, equivalently, the "
        "number of probabilities at each time step mini_batch How many "
        "examples in a minibatch. "
        "\n"
        "info see struct ctcOptions"
        "\n"
        "Return a pair [status, size_in_bytes], where size_in_bytes the is "
        "memory requirements in bytes; This memory should be allocated at the "
        "same place, CPU or GPU, that the probs are in",
        py::arg("label_lengths"), py::arg("input_lengths"),
        py::arg("alphabet_size"), py::arg("minibatch"), py::arg("info"));
}
