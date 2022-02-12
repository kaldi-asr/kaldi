// chainbin/nnet3-chain-add-post-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2021       Behavox (author: Hossein Hadian)

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
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-chain-training.h"
#include "cudamatrix/cu-allocator.h"

#include "chain/chain-training.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-generic-numerator.h"
#include "chain/chain-denominator.h"

namespace kaldi {

void Clip(Matrix<BaseFloat> &mat, BaseFloat threshold) {
  for (int32 i = 0; i < mat.NumRows(); i++) {
    for (int32 j = 0; j < mat.NumCols(); j++) {
      if (abs(mat(i, j)) < threshold)
        mat(i, j) = 0.0;
    }
  }
}

namespace chain {
void ComputeChainOccupancies(const ChainTrainingOptions &opts,
                             const DenominatorGraph &den_graph,
                             const Supervision &supervision,
                             const CuMatrixBase<BaseFloat> &nnet_output,
                             bool do_num,
                             BaseFloat den_scale,
                             CuMatrix<BaseFloat> *gamma) {
  bool denominator_ok = true;
  bool numerator_ok = true;
  gamma->Resize(nnet_output.NumRows(), nnet_output.NumCols(), kSetZero);

  if (den_scale != 0.0) {
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);
    BaseFloat den_logprob_weighted = supervision.weight * denominator.Forward();
    denominator_ok = denominator.Backward(den_scale, gamma);
    if (!denominator_ok) KALDI_ERR << "den failed";
  }

  if (do_num) {
    GenericNumeratorComputation numerator(opts.numerator_opts,
                                          supervision, nnet_output);
    BaseFloat num_logprob_weighted;
    numerator_ok = numerator.ForwardBackward(&num_logprob_weighted, gamma);
    if (!numerator_ok) KALDI_ERR << "num failed";
  }
}

} // namespace chain
}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program reads the input nnet3 egs, computes nnet outputs for them and .\n"
        "appends the outputs as NnetIos to the egs. For Continual Learning. Search for LWF.\n"
        "Usage:  nnet3-chain-add-post-to-egs [options] <nnet-chain- model> <denominator-fst> \
<egs-rspecifier> <egs-wspecifier> \n";

    NnetChainTrainingOptions opts;
    bool batchnorm_test_mode = false, dropout_test_mode = false;

    std::string use_gpu = "yes", type = "raw";
    BaseFloat clip_threshold = 0.0;
    bool apply_exp = true;

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("type", &type,
                "raw|den, what type of posts to append.");
    po.Register("clip-threshold", &clip_threshold, "Threshold to clip posterior values");
    po.Register("apply-exp", &apply_exp, "Apply exp to the posteriors.");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    opts.Register(&po);
#if HAVE_CUDA==1
    CuDevice::RegisterDeviceOptions(&po);
#endif
    RegisterCuAllocatorOptions(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        den_fst_rxfilename = po.GetArg(2);
    std::string examples_rspecifier = po.GetArg(3),
        examples_wspecifier = po.GetArg(4);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

    CachingOptimizingCompiler compiler(nnet, opts.nnet_config.optimize_config,
                                       opts.nnet_config.compiler_config);
    const NnetTrainerOptions &nnet_config = opts.nnet_config;

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);
    int32 num_pdfs = nnet.OutputDim("output");
    if (num_pdfs < 0) {
      KALDI_ERR << "Neural net '" << nnet_rxfilename
                << "' has no output named 'output'";
    }
    chain::DenominatorGraph den_graph(den_fst, num_pdfs);

    int64 num_read = 0, num_written = 0, num_err = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      const std::string &key = example_reader.Key();
      NnetChainExample eg(example_reader.Value());

      bool need_model_derivative = false;
      bool use_xent_regularization = (opts.chain_config.xent_regularize != 0.0);
      ComputationRequest request;
      GetChainComputationRequest(nnet, eg, need_model_derivative,
                                 nnet_config.store_component_stats,
                                 use_xent_regularization, need_model_derivative,
                                 &request);
      std::shared_ptr<const NnetComputation> computation = compiler.Compile(request);
      NnetComputer computer(nnet_config.compute_config, *computation,
                            &nnet, NULL);
      computer.AcceptInputs(nnet, eg.inputs);
      computer.Run();
      const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
      const NnetChainSupervision &sup = eg.outputs[0];  // assume only one output

      if (type == "raw") {
        Matrix<BaseFloat> full_post(nnet_output);
        if (apply_exp) {
          KALDI_VLOG(1) << "Adding exped full raw post to " << key << ".";
          full_post.ApplyExp();
        } else {
          KALDI_VLOG(1) << "Adding full raw post to " << key << ".";
        }
        NnetIo full_io("__LWF-posteriors", 0, full_post, 3);
        eg.inputs.push_back(full_io);
      } else if (type == "den") {
        CuMatrix<BaseFloat> den_gamma;
        ComputeChainOccupancies(opts.chain_config, den_graph, sup.supervision, nnet_output,
                                false, 1.0, &den_gamma);
        Matrix<BaseFloat> full_post(den_gamma);
        Clip(full_post, clip_threshold);
        KALDI_VLOG(1) << "Adding den only post to " << key << ".";
        NnetIo full_io("__LWF-posteriors", 0, full_post, 3);
        eg.inputs.push_back(full_io);
      }
      example_writer.Write(key, eg);
      num_written++;
    }
    KALDI_LOG << "Read " << num_read
              << " neural-network training examples, wrote " << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
