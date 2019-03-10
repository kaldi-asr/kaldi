// rnnlmbin/rnnlm-train.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)

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
#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "nnet3/nnet-utils.h"
#include "cudamatrix/cu-allocator.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;


    // rnnlm_rxfilename must be supplied, via --read-rnnlm option.
    std::string rnnlm_rxfilename;
    // For now, rnnlm_wxfilename must be supplied (later we could make it possible
    // to train the embedding matrix without training the RNNLM itself, if there
    // is a need).
    std::string rnnlm_wxfilename;
    // embedding_rxfilename must be supplied, via --read-embedding option.
    std::string embedding_rxfilename;
    std::string embedding_wxfilename;
    std::string word_features_rxfilename;
    // binary mode for writing output.
    bool binary = true;

    RnnlmCoreTrainerOptions core_config;
    RnnlmEmbeddingTrainerOptions embedding_config;
    RnnlmObjectiveOptions objective_config;

    const char *usage =
        "Train nnet3-based RNNLM language model (reads minibatches prepared\n"
        "by rnnlm-get-egs).  Supports various modes depending which parameters\n"
        "we are training.\n"
        "Usage:\n"
        " rnnlm-train [options] <egs-rspecifier>\n"
        "e.g.:\n"
        " rnnlm-get-egs ... ark:- | \\\n"
        " rnnlm-train --read-rnnlm=foo/0.raw --write-rnnlm=foo/1.raw --read-embedding=foo/0.embedding \\\n"
        "       --write-embedding=foo/1.embedding --read-sparse-word-features=foo/word_feats.txt ark:-\n"
        "See also: rnnlm-get-egs\n";


    std::string use_gpu = "yes";

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("read-rnnlm", &rnnlm_rxfilename,
                "Read RNNLM from this location (e.g. 0.raw).  Must be supplied.");
    po.Register("write-rnnlm", &rnnlm_wxfilename,
                "Write RNNLM to this location (e.g. 1.raw)."
                "If not supplied, the core RNNLM is not trained "
                "(but other parts of the model might be.");
    po.Register("read-embedding", &embedding_rxfilename,
                "Location to read dense (feature or word) embedding matrix, "
                "of dimension (num-words or num-features) by (embedding-dim).");
    po.Register("write-embedding", &embedding_wxfilename,
                "Location to write embedding matrix (c.f. --read-embedding). "
                "If not provided, the embedding will not be trained.");
    po.Register("read-sparse-word-features", &word_features_rxfilename,
                "Location to read sparse word-feature matrix, e.g. "
                "word_feats.txt.  Format is lines like: '1  30 1.0 516 1.0':"
                "starting with word-index, then a list of pairs "
                "(feature-index, value) only including nonzero features. "
                "This will usually be determined in an ad-hoc way based on "
                "letters and other hand-built features; it's not trainable."
                " If present, the embedding matrix read via --read-embedding "
                "will be interpreted as a feature-embedding matrix.");
    po.Register("binary", &binary,
                "If true, write outputs in binary form.");


    objective_config.Register(&po);
    RegisterCuAllocatorOptions(&po);

    // register the core RNNLM training options options with the prefix "rnnlm",
    // so they will appear as --rnnlm.max-change and the like.  This is done
    // with a prefix because later we may add a neural net to transform the word
    // embedding, and it would have options that would have a name conflict with
    // some of these options.
    ParseOptions core_opts("rnnlm", &po);
    core_config.Register(&core_opts);
    // ... and register the embedding options with the prefix "embedding".
    ParseOptions embedding_opts("embedding", &po);
    embedding_config.Register(&embedding_opts);




    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }
    if (rnnlm_rxfilename == "" ||
        rnnlm_wxfilename == "" ||
        embedding_rxfilename == "") {
      KALDI_WARN << "--read-rnnlm, --write-rnnlm and --read-embedding "
          "options are required.";
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;

    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    if (!IsSimpleNnet(rnnlm))
      KALDI_ERR << "Input RNNLM in " << rnnlm_rxfilename
                << " is not the type of neural net we were looking for; "
          "failed IsSimpleNnet().";

    CuMatrix<BaseFloat> embedding_mat;
    ReadKaldiObject(embedding_rxfilename, &embedding_mat);

    CuSparseMatrix<BaseFloat> word_feature_mat;

    if (word_features_rxfilename != "") {
      // binary mode is not supported here; it's a text format.
      Input input(word_features_rxfilename);
      int32 feature_dim = embedding_mat.NumRows();
      SparseMatrix<BaseFloat> cpu_word_feature_mat;
      ReadSparseWordFeatures(input.Stream(), feature_dim,
                             &cpu_word_feature_mat);
      word_feature_mat.Swap(&cpu_word_feature_mat);  // copy to GPU, if we have
                                                     // one.
    }


    {
      bool train_embedding = (embedding_wxfilename != "");

      RnnlmTrainer trainer(
          train_embedding, core_config, embedding_config, objective_config,
          (word_features_rxfilename != "" ? &word_feature_mat : NULL),
          &embedding_mat, &rnnlm);

      SequentialRnnlmExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next())
        trainer.Train(&(example_reader.Value()));

      if (trainer.NumMinibatchesProcessed() == 0)
        KALDI_ERR << "There was no data to train on.";
      // The destructor of 'trainer' trains on the last minibatch
      // and writes out anything we need to write out.
    }

    WriteKaldiObject(rnnlm, rnnlm_wxfilename, binary);
    KALDI_LOG << "Wrote RNNLM to "
              << PrintableWxfilename(rnnlm_wxfilename);
    if (embedding_wxfilename != "") {
      WriteKaldiObject(embedding_mat, embedding_wxfilename, binary);
      KALDI_LOG << "Wrote embedding matrix to "
                << PrintableWxfilename(embedding_wxfilename);
    }


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
