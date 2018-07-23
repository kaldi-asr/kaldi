// nnet3bin/nnet3-discriminative-compute-from-egs.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-discriminative-example.h"
#include "nnet3/nnet-optimize.h"


namespace kaldi {
namespace nnet3 {

class NnetComputerFromEg {
 public:
  NnetComputerFromEg(const Nnet &nnet):
      nnet_(nnet), compiler_(nnet) { }

  // Compute the output (which will have the same number of rows as the number
  // of Indexes in the output of the eg), and put it in "output".
  void Compute(const NnetExample &eg, Matrix<BaseFloat> *output) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false;
    GetComputationRequest(nnet_, eg, need_backprop, store_stats, &request);
    const NnetComputation &computation = *(compiler_.Compile(request));
    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);
    computer.AcceptInputs(nnet_, eg.io);
    computer.Run();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    output->Resize(nnet_output.NumRows(), nnet_output.NumCols());
    nnet_output.CopyToMat(output);
  }
 private:
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;

};

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Read input nnet discriminative training examples, and compute the "
        "output for each one. This program is similar to "
        "nnet3-compute-from-egs, but works with discriminative egs. \n"
        "If --apply-exp=true, apply the Exp() function to the output before writing\n"
        "it out.\n"
        "Note: This program uses only the input; it does not do forward-backward\n"
        "over the lattice. See nnet3-discriminative-compute-objf for that.\n"
        "\n"
        "Usage:  nnet3-discriminative-compute-from-egs [options] <raw-nnet-in> <training-examples-in> <matrices-out>\n"
        "e.g.:\n"
        "nnet3-discriminative-compute-from-egs --apply-exp=true 0.raw ark:1.degs ark:- | matrix-sum-rows ark:- ... \n"
        "See also: nnet3-compute nnet3-compute-from-egs\n";

    bool binary_write = true,
        apply_exp = false;
    std::string use_gpu = "yes";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("apply-exp", &apply_exp, "If true, apply exp function to "
                "output");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetComputerFromEg computer(nnet);

    int64 num_egs = 0;

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    for (; !example_reader.Done(); example_reader.Next(), num_egs++) {
      Matrix<BaseFloat> output;
      NnetExample eg;
      NnetDiscriminativeExample disc_eg = example_reader.Value();
      eg.io.swap(disc_eg.inputs);

      for (int32 i = 0; i < disc_eg.outputs.size(); i++) {
        NnetIo io;
        io.name = disc_eg.outputs[i].name;
        io.indexes = disc_eg.outputs[i].indexes;
        eg.io.push_back(io);
      }

      computer.Compute(eg, &output);
      KALDI_ASSERT(output.NumRows() != 0);
      if (apply_exp)
        output.ApplyExp();
      matrix_writer.Write(example_reader.Key(), output);
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Processed " << num_egs << " examples.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
