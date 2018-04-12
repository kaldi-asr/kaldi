// nnet3bin/nnet3-xvector-compute.cc

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder

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

namespace kaldi {
namespace nnet3 {

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
    const Nnet &nnet, CachingOptimizingCompiler *compiler,
    Vector<BaseFloat> *xvector) {
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;
  request.inputs.push_back(
    IoSpecification("input", 0, features.NumRows()));
  IoSpecification output_spec;
  output_spec.name = "output";
  output_spec.has_deriv = false;
  output_spec.indexes.resize(1);
  request.outputs.resize(1);
  request.outputs[0].Swap(&output_spec);
  std::shared_ptr<const NnetComputation> computation = compiler->Compile(request);
  Nnet *nnet_to_update = NULL;  // we're not doing any update.
  NnetComputer computer(NnetComputeOptions(), *computation,
                  nnet, nnet_to_update);
  CuMatrix<BaseFloat> input_feats_cu(features);
  computer.AcceptInput("input", &input_feats_cu);
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  xvector->Resize(cu_output.NumCols());
  xvector->CopyFromVec(cu_output.Row(0));
}

} // namespace nnet3
} // namespace kaldi

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
        "Usage: nnet3-xvector-compute [options] <raw-nnet-in> "
        "<features-rspecifier> <vector-wspecifier>\n"
        "e.g.: nnet3-xvector-compute final.raw scp:feats.scp "
        "ark:nnet_prediction.ark\n"
        "See also: nnet3-compute\n";

    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    CachingOptimizingCompilerOptions compiler_config;

    opts.acoustic_scale = 1.0; // by default do no scaling in this recipe.

    std::string use_gpu = "no";
    int32 chunk_size = -1,
      min_chunk_size = 100;

    opts.Register(&po);
    compiler_config.Register(&po);

    po.Register("use-gpu", &use_gpu,
      "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("chunk-size", &chunk_size,
      "If set, extracts xectors from specified chunk-size, and averages.  "
      "If not set, extracts an xvector from all available features.");
    po.Register("min-chunk-size", &min_chunk_size,
      "Minimum chunk-size allowed when extracting xvectors.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                vector_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    CachingOptimizingCompiler compiler(nnet, opts.optimize_config, compiler_config);

    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;
    int32 xvector_dim = nnet.OutputDim("output");

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      int32 num_rows = features.NumRows(),
            feat_dim = features.NumCols(),
            this_chunk_size = chunk_size;

      if (num_rows < min_chunk_size) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                   << " is greater than the number of rows "
                   << "in utterance: " << utt;
        num_fail++;
        continue;
      } else if (num_rows < chunk_size) {
        KALDI_LOG << "Chunk size of " << chunk_size << " is greater than "
                  << "the number of rows in utterance: " << utt
                  << ", using chunk size  of " << num_rows;
        this_chunk_size = num_rows;
      } else if (chunk_size == -1) {
        this_chunk_size = num_rows;
      }

      int32 num_chunks = ceil(
        num_rows / static_cast<BaseFloat>(this_chunk_size));
      Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
      BaseFloat tot_weight = 0.0;

      // Iterate over the feature chunks.
      for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        // If we're nearing the end of the input, we may need to shift the
        // offset back so that we can get this_chunk_size frames of input to
        // the nnet.
        int32 offset = std::min(
          this_chunk_size, num_rows - chunk_indx * this_chunk_size);
        if (offset < min_chunk_size)
          continue;
        SubMatrix<BaseFloat> sub_features(
          features, chunk_indx * this_chunk_size, offset, 0, feat_dim);
        Vector<BaseFloat> xvector;
        tot_weight += offset;
        RunNnetComputation(sub_features, nnet, &compiler, &xvector);
        xvector_avg.AddVec(offset, xvector);
      }
      xvector_avg.Scale(1.0 / tot_weight);
      vector_writer.Write(utt, xvector_avg);

      frame_count += features.NumRows();
      num_success++;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
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
