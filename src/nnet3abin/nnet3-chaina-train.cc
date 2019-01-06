// nnet3bin/nnet3-chaina-train.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3a/nnet-chaina-training.h"
#include "cudamatrix/cu-allocator.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+chaina (i.e. chain + adaptation framework) neural network.\n"
        "Minibatches are to be created by nnet3-chain-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-chaina-train [options] <model-in-dir> <den-fst-dir> <transform-dir>\n"
        "        <egs-rspecifier>  <model-out-dir>\n"
        "\n"
        "<model-in-dir> should contain bottom.raw, and <lang>.mdl for each language <lang>\n"
        "<den-fst-dir> should contain <lang>.den.fst for each language <lang>\n"
        "<transform-dir> should contain <lang>.ada for each language <lang>\n"
        "<model-out-dir> is a place to where bottom.raw and <lang>.raw for each language\n"
        "  <lang> that was seen in the egs, will be written.\n";


    int32 srand_seed = 0;
    bool binary_write = true;
    std::string use_gpu = "yes";
    NnetChainaTrainingOptions chaina_opts;
    int32 job_id = 0;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("job-id", &job_id,
                "Job identifier, helps to determine pathnames of models written "
                "to <model-out-dir>.");

    chaina_opts.Register(&po);
    RegisterCuAllocatorOptions(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    bool ok;

    std::string model_in_dir = po.GetArg(1),
        den_fst_dir = po.GetArg(2),
        transform_dir = po.GetArg(3),
        egs_rspecifier = po.GetArg(4),
        model_out_dir = po.GetArg(5);

    NnetChainaModels models(chaina_opts.nnet_config.zero_component_stats,
                            chaina_opts.bottom_model_test_mode,
                            chaina_opts.top_model_test_mode,
                            model_in_dir, den_fst_dir, transform_dir);

    {
      NnetChainaTrainer trainer(chaina_opts, &models);

      SequentialNnetChainExampleReader example_reader(egs_rspecifier);

      for (; !example_reader.Done(); example_reader.Next())
        trainer.Train(example_reader.Key(),
                      example_reader.Value());

      ok = trainer.PrintTotalStats();
    }
    models.WriteRawModels(model_out_dir, binary_write, job_id);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
