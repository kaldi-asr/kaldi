// nnet3bin/nnet3-compute.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2016   David Snyder

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
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "xvector/nnet-xvector-compute.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Propagate the features through the network and write the output\n"
      "xvectors.  By default, xvectors are extracted once every\n"
      "--xvector-period using --chunk-size frames and output as an archive\n"
      "of matrices.  If --repeat=true, the xvectors are copied between\n"
      "periods, so that the output matrix has the same number of rows as\n"
      "the input.  If --output-as-vector=true, the xvectors are averaged\n"
      "across periods, and the output is a single vector for each utterance.\n"
      "\n"
      "Usage: nnet3-xvector-compute [options] <raw-nnet-in> "
      "<feats-rspecifier> <xvector-wspecifier>\n"
      " e.g.: nnet3-xvector-compute --xvector-period=50 final.raw "
      "scp:feats.scp ark:xvectors.ark\n";

    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    std::string use_gpu = "yes";
    int32 chunk_size = 100;

    opts.Register(&po);

    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("chunk-size", &chunk_size,
      "Feature chunk size over which the xvector is computed.  "
      "If not set, defaults to xvector-period.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    KALDI_ASSERT(chunk_size > 0);

    std::string nnet_rxfilename = po.GetArg(1),
                feat_rspecifier = po.GetArg(2),
                vector_wspecifier = po.GetArg(3);
    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    NnetXvectorComputer nnet_computer(opts, &nnet);

    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_success = 0,
          num_fail = 0,
          left_context,
          right_context,
          xvector_dim = nnet.OutputDim("output");
    int32 min_chunk_size = 100;
    int64 frame_count = 0;
    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats (feat_reader.Value());
      int32 num_rows = feats.NumRows(),
            feat_dim = feats.NumCols(),
            this_chunk_size = chunk_size;

      if (num_rows < min_chunk_size) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                   << " is greater than the number of rows "
                   << "in utterance: " << utt;
        num_fail++;
        continue;
      } else if (num_rows < this_chunk_size) {
        KALDI_LOG << "Chunk size of " << this_chunk_size << " is greater than "
                  << "the number of rows in utterance: " << utt
                  << ", using chunk size  of " << num_rows;
        this_chunk_size = num_rows;
      }

      int32 num_chunks = ceil(num_rows / static_cast<BaseFloat>(chunk_size));

      Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
      BaseFloat tot_weight = 0.0;

      // Iterate over the feature chunks.
      for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        // If we're nearing the end of the input, we may need to shift the
        // offset back so that we can get this_chunk_size frames of input to
        // the nnet.
        int32 offset = std::min(chunk_size, num_rows - chunk_indx * chunk_size);
        if (offset < min_chunk_size)
          continue;
        SubMatrix<BaseFloat> sub_feats(feats, chunk_indx * chunk_size, offset,
                                       0, feat_dim);
        Vector<BaseFloat> xvector(xvector_dim);
        nnet_computer.ComputeXvector(sub_feats, &xvector);
        tot_weight += offset;
        xvector_avg.AddVec(offset, xvector);
      }

      // If output is a vector, scale it by the total weight.
      xvector_avg.Scale(1.0 / tot_weight);
      vector_writer.Write(utt, xvector_avg);

      frame_count += feats.NumRows();
      num_success++;
    }

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
