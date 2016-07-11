// nnet3bin/nnet3-train.cc

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
#include "nnet3/nnet-training.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

// Initializes the trainer with recurrent node names and their offsets.
// It is used for state preserving training. But if state preserving traning
// mode is off, it has no effect on the trainer
void InitializeForStatePreservingTraining(const Nnet &nnet, NnetTrainer *trainer,
                                          std::vector<std::string>
                                          *recurrent_output_names) {
  // extracts recurrent output names and their offsets from nnet 
  std::vector<std::string> recurrent_node_names; 
  std::vector<int32> recurrent_offsets;
  GetRecurrentOutputNodeNames(nnet, recurrent_output_names,
                              &recurrent_node_names);
  GetRecurrentNodeOffsets(nnet, recurrent_node_names, &recurrent_offsets);
  for (int32 i = 0; i < recurrent_offsets.size(); i++)
    KALDI_VLOG(2) << "recurrent node: " << recurrent_node_names[i]
                  << ", offset: " << recurrent_offsets[i];

  // passes on the above info to trainer
  trainer->GiveStatePreservingInfo(*recurrent_output_names, recurrent_offsets);
}

// Updates the recurrent inputs of the current minibatch with the recurrent
// outputs of the previous minibatch. We copy the current minibatch eg_in to
// *eg_out and then update *eg_out. It is used for state preserving training,
// and it is called just before actual training.
void UpdateMinibatch(const NnetTrainer &trainer, const NnetExample &eg_in,
                     const std::vector<std::string> &recurrent_output_names,
                     NnetExample *eg_out) {
  *eg_out = eg_in;
  for (int32 i = 0; i < recurrent_output_names.size(); i++) {
    const std::string &node_name = recurrent_output_names[i];
    for (int32 f = 0; f < eg_out->io.size(); f++) {
      NnetIo &io = eg_out->io[f];
      if (io.name == (node_name + "_STATE_PREVIOUS_MINIBATCH")) {
        const CuMatrix<BaseFloat> &recurrent_output =
            trainer.GetRecurrentOutput(i);
        KALDI_ASSERT(io.features.NumRows() == recurrent_output.NumRows() &&
                     io.features.NumCols() == recurrent_output.NumCols());
        io.features = recurrent_output;
        break;
      }
    }
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3 neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU); see nnet3-train-parallel for multi-threaded training\n"
        "that is better suited to CPUs.\n"
        "\n"
        "Usage:  nnet3-train [options] <raw-model-in> <training-examples-in> <raw-model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet3-train 1.raw 'ark:nnet3-merge-egs 1.egs ark:-|' 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    int32 num_minibatches_per_chunk = 1;
    NnetTrainerOptions train_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("num-minibatches-per-chunk", &num_minibatches_per_chunk,
                "number of new, smaller chunks after splitting an original "
                "chunk. Each of these new chunks is in a new minibatch, which "
                "is why it is named in this way. If > 1, then state preserving "
                "training will be enabled.");

    train_config.Register(&po);

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
        nnet_wxfilename = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetTrainer trainer(train_config, &nnet);

    std::vector<std::string> recurrent_output_names;
    InitializeForStatePreservingTraining(nnet, &trainer,
                                         &recurrent_output_names);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    for (int32 count = 0; !example_reader.Done(); example_reader.Next()) {
      if (num_minibatches_per_chunk > 1 && count > 0) {
        NnetExample eg;
        Timer tim;//debug
        UpdateMinibatch(trainer, example_reader.Value(),
                        recurrent_output_names, &eg);
        KALDI_LOG << "update-minibatch time: "<< tim.Elapsed();//debug
        trainer.Train(eg);
      } else
        trainer.Train(example_reader.Value());
      count = (count + 1) % num_minibatches_per_chunk;
    }

    bool ok = trainer.PrintTotalStats();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Wrote model to " << nnet_wxfilename;
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


