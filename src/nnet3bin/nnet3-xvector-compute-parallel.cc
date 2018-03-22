// nnet3bin/nnet3-xvector-compute-parallel.cc

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder
//           2018   Behavox Limited (author: Arseniy Gorin)

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
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-xvector-threaded.h"
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Propagate features through an xvector neural network model and write\n"
        "the output vectors.  \"Xvector\" is our term for a vector or\n"
        "embedding which is the output of a particular type of neural network\n"
        "architecture found in speaker recognition.  This architecture\n"
        "consists of several layers that operate on frames, a statistics\n"
        "pooling layer that aggregates over the frame-level representations\n"
        "and possibly additional layers that operate on segment-level\n"
        "representations.  The xvectors are generally extracted from an\n"
        "output layer after the statistics pooling layer.  By default, one\n"
        "xvector is extracted directly from the set of features for each\n"
        "utterance.  Optionally, xvectors are extracted from chunks of input\n"
        "features and averaged, to produce a single vector.\n"
        "\n"
        "This version supports multi threading. WARNING: this works properly\n"
        "only if you use <compiler-cache> with the same parameters \n"
        "(to not allow re-compiling cache in multiple thread, as this operation is not thread-safe)\n"
        "\n" 
        "Usage: nnet3-xvector-compute [options] <raw-nnet-in> "
        "<features-rspecifier> <vector-wspecifier>\n"
        "e.g.: nnet3-xvector-compute <compiler-cache> final.raw scp:feats.scp "
        "ark:nnet_prediction.ark\n"
        "See also: nnet3-compute\n";

    ParseOptions po(usage);
    Timer timer;

    TaskSequencerConfig sequencer_config; // has --num-threads option
    NnetSimpleComputationOptions opts;

    opts.acoustic_scale = 1.0; // by default do no scaling in this recipe.

    int32 chunk_size = -1,
      min_chunk_size = 100,
      chunk_sampling_rate = 1;

    opts.Register(&po);
    sequencer_config.Register(&po);

    CachingOptimizingCompilerOptions compiler_config; 
    compiler_config.Register(&po);
    po.Register("chunk-size", &chunk_size,
      "If set, extracts xectors from specified chunk-size, and averages.  "
      "If not set, extracts an xvector from all available features.");
    po.Register("min-chunk-size", &min_chunk_size,
      "Minimum chunk-size allowed when extracting xvectors.");
    po.Register("chunk-sampling-rate", &chunk_sampling_rate,
      "Chunk size will be rounded to this number of frames (to take advantage of compiler cache).");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string cache_rxfilename = "", nnet_rxfilename, feature_rspecifier, vector_wspecifier;
   
    if (po.NumArgs() == 3) {
      nnet_rxfilename = po.GetArg(1);
      feature_rspecifier = po.GetArg(2);
      vector_wspecifier = po.GetArg(3);
    }
    else {
      cache_rxfilename = po.GetArg(1);
      nnet_rxfilename = po.GetArg(2);
      feature_rspecifier = po.GetArg(3);
      vector_wspecifier = po.GetArg(4);
    }

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;

    TaskSequencer<XVectorExtractorParallelClass> sequencer(sequencer_config);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    if (chunk_size > 0 and chunk_sampling_rate > 0) {
        compiler_config.cache_capacity = (chunk_size - min_chunk_size) / chunk_sampling_rate + 1;
    }
    CachingOptimizingCompiler compiler(nnet, opts.optimize_config, compiler_config);

    if (cache_rxfilename != "") {
        KALDI_LOG << "Reading cache from " << cache_rxfilename;
        bool cache_binary_in;
        Input ki(cache_rxfilename, &cache_binary_in);
        compiler.ReadCache(ki.Stream(), cache_binary_in);
    }

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> &features (feature_reader.Value());

      // pad features to make sure chunk_sampling_rate is satisfied
      int32 num_rows = features.NumRows(),
            feat_dim = features.NumCols();
     
      if (num_rows < min_chunk_size) { 
          KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                     << " is greater than the number of rows "
                     << "in utterance: " << utt;
          num_fail++;
          continue;
      }
 
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }

      if (chunk_sampling_rate > 1) {
          int32 feat_pad = chunk_sampling_rate - num_rows % chunk_sampling_rate;
          if (feat_pad > 0){
              features.Resize(num_rows + feat_pad, feat_dim, kCopyData);
              for (int32 i = 0; i < feat_pad; i++) {
                for (int32 j = 0; j < feat_dim; j++) {
                  features(num_rows + i, j) = features(i, j);
                }
              }
          }
      }

      sequencer.Run(new XVectorExtractorParallelClass(opts, nnet, &compiler, utt, chunk_size, min_chunk_size, 
                                             features, &vector_writer));
      frame_count += features.NumRows();
      num_success++;
    }
    sequencer.Wait(); 
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
