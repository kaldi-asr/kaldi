// nnet3bin/nnet3-chaina-combine.cc

// Copyright 2019  Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {
namespace nnet3 {


/**
   Computes the average objective function of the provided egs with the provided
   set of models.
     @param [in] opts  The options class for the objective computation
                (shares the same option as training, but we set the training
                options to true)
     @param [in] unadapted_objf_weight  A number in the range [0,1] that says
                how much weight we put on the unadapted version of the
                objective function when choosing models.
     @param [in] num_models_averaged   Needed only for diagnostics-- the
                number of sets of models that we averaged to get the
                models in the 'models' object
     @param [in] keys_and_egs  The vector containing the examples we
                are to evaluate the objective function on, and the corresponding
                string-valued keys (needed because the language name and
                example weight are optionally encoded in it).
     @param [in,out] models  The models that we are evaluating the objective
               function.  These will only be modified to to the extent that
               the batchnorm stats and any component-level stats would be
               affected.
 */
BaseFloat GetObjectiveFunction(
    const NnetChainaTrainingOptions &opts,
    BaseFloat unadapted_objf_weight,
    int32 num_models_averaged,
    const std::vector<std::pair<std::string, NnetChainExample> >& keys_and_egs,
    NnetChainaModels *models) {
  KALDI_ASSERT(!opts.top.train && !opts.bottom.train);
  NnetChainaTrainer trainer(opts, models);
  size_t num_egs = keys_and_egs.size();
  for (size_t i = 0; i < num_egs; i++) {
    trainer.Train(keys_and_egs[i].first, keys_and_egs[i].second);
  }
  BaseFloat weight, adapted_objf, unadapted_objf;
  adapted_objf = trainer.GetTotalObjf(true, &weight);
  adapted_objf /= weight;
  unadapted_objf = trainer.GetTotalObjf(false, &weight);
  unadapted_objf /= weight;
  BaseFloat ans = unadapted_objf_weight * unadapted_objf +
      (1.0 - unadapted_objf_weight) * adapted_objf;
  KALDI_LOG << "When averaging " << num_models_averaged
            << " models, objf values (unadapted/si,adapted) "
            << unadapted_objf << ", " << adapted_objf
            << ", interpolated = " << ans << "; over "
            << weight << " frames.";
  return ans;
}

void ReadExamples(
    const std::string &egs_rspecifier,
    std::vector<std::pair<std::string, NnetChainExample> > *keys_and_egs) {
  keys_and_egs->reserve(10000);  // reserve a lot of space to minimize the chance of
  // reallocation.
  SequentialNnetChainExampleReader example_reader(egs_rspecifier);
  for (; !example_reader.Done(); example_reader.Next()) {
    size_t i = keys_and_egs->size();
    keys_and_egs->resize(i + 1);
    keys_and_egs->back().first = example_reader.Key();
    keys_and_egs->back().second.Swap(&(example_reader.Value()));
  }
  KALDI_LOG << "Read " << keys_and_egs->size() << " examples.";
  KALDI_ASSERT(!keys_and_egs->empty());
}


}  // namespace nnet3
}  // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program does the final model-combination stage of 'chaina'\n"
        "acoustic training: it averages over the last n models, where the\n"
        "'n' is chosen (by this program) based on maximizing the objective\n"
        "function on the data given to it.  It maximizes the average of the\n"
        "speaker-independent and speaker-dependent versions of the 'chain'\n"
        "objective values.\n"
        "This program is intended to be used with a GPU.\n"
        "\n"
        "Usage:  nnet3-chaina-combine [options] <model1-in-dir> ... <modelN-in-dir> \\\n"
        "    <den-fst-dir> <transform-dir> <egs-rspecifier> <model-out-dir>\n"
        "\n"
        "<modelX-in-dir> should contain bottom.raw, and <lang>.mdl for each language <lang>\n"
        " (these will be averaged over a range of indexes including N, e.g. just modelN, or\n"
        "  modelN with model(N-1), and so on).\n"
        "<den-fst-dir> should contain <lang>.den.fst for each language <lang>\n"
        "<transform-dir> should contain <lang>.ada for each language <lang>\n"
        "<model-out-dir> is a place to where bottom.mdl and <lang>.mdl for each language\n"
        "  <lang> that was seen in the egs, will be written (for <job-id>, see the --job-id option).\n";


    int32 srand_seed = 0;
    bool binary_write = true;
    std::string use_gpu = "yes";
    NnetChainaTrainingOptions chaina_opts;
    chaina_opts.top.train = false;
    chaina_opts.bottom.train = false;
    chaina_opts.top.dropout_test_mode = true;
    chaina_opts.bottom.dropout_test_mode = true;
    // But leave the batchnorm test-modes at false.

    // Setting batchnorm_stats_scale to 1.0 means it won't scale down the
    // batchnorm stats as it goes (the default is 0.8), so they will all be
    // remembered.  Note: each time we initialize and use the trainer object, in
    // GetObjectiveFunction, it will call ZeroComponentStats() for both the
    // bottom and top models (assuming the options are the defaults), so only
    // the stats from the most recent run will be present.
    chaina_opts.nnet_config.batchnorm_stats_scale = 1.0;

    BaseFloat unadapted_objf_weight = 0.5;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("unadapted-weight", &unadapted_objf_weight,
                "The weight we give to the unadapted version of the objective function "
                "when evaluating the goodness of models (the adapted objective gets "
                "1 minus this value as its weight)");


    chaina_opts.Register(&po);
    RegisterCuAllocatorOptions(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    int32 n = po.NumArgs() - 4;  // n is the number of models we have
                                 // available to average.

    std::string last_model_in_dir = po.GetArg(n),
        den_fst_dir = po.GetArg(n + 1),
        transform_dir = po.GetArg(n + 2),
        egs_rspecifier = po.GetArg(n + 3),
        model_out_dir = po.GetOptArg(n + 4);

    NnetChainaModels models(chaina_opts,
                            last_model_in_dir, den_fst_dir,
                            transform_dir);


    std::vector<std::pair<std::string, NnetChainExample> > keys_and_egs;
    ReadExamples(egs_rspecifier, &keys_and_egs);

    // first evaluates the objective using the last model.
    int32 best_num_to_combine = -1;
    BaseFloat best_objf = -std::numeric_limits<BaseFloat>::infinity(),
                          single_model_objf;

    std::unique_ptr<NnetChainaModels> best_models;

    for (int32 num_models = 1; num_models <= n; num_models++) {
      if (num_models > 1)
        models.InterpolateWith(1.0 / num_models, po.GetArg(n + 1 - num_models));
      BaseFloat objf = GetObjectiveFunction(chaina_opts,  unadapted_objf_weight,
                                            num_models, keys_and_egs, &models);
      if (objf > best_objf || num_models == 1) {
        best_objf = objf;
        best_models = std::unique_ptr<NnetChainaModels>(
            new NnetChainaModels(models));
        best_num_to_combine = num_models;
        if (num_models == 1)
          single_model_objf = objf;
      }
      if (num_models > best_num_to_combine + 4 && num_models < n)
        KALDI_LOG << "Stopping the search early as it looks like we found "
            "the best combination";
    }

    KALDI_LOG << "Best objective function was " << best_objf << " with "
              << best_num_to_combine << " models.";
    KALDI_LOG << "About to recompute objective function with batchnorm in "
        "test-mode:\n";
    chaina_opts.top.batchnorm_test_mode = true;
    chaina_opts.bottom.batchnorm_test_mode = true;

    BaseFloat test_mode_objf =
        GetObjectiveFunction(chaina_opts, unadapted_objf_weight,
                             best_num_to_combine,
                             keys_and_egs,
                             best_models.get());
    KALDI_LOG << "Objf with test-mode batchnorm was " << test_mode_objf
              << " (vs. " << best_objf << " without test mode)";

    KALDI_LOG << "Combination changed the objective from "
              << single_model_objf << " with only the final model, to "
              << best_objf << " with " << best_num_to_combine
              << " models.";

    best_models->WriteCombinedModels(model_out_dir, binary_write);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
