// online2bin/apply-cmvn-online.cc

// Copyright      2014  Johns Hopkins University (author: Daniel Povey)

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

#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/online-feature.h"
#include "cudafeat/feature-online-cmvn-cuda.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
      "Apply online cepstral mean (and possibly variance) computation online,\n"
      "using the same code as used for online decoding in the 'new' setup in\n"
      "online2/ and online2bin/.'\n"
      "The computation is done on the device in serial. " 
      "spk2utt is not supported.\n"
      "\n"
      "Usage: apply-cmvn-online-cuda [options] <global-cmvn-stats> <feature-rspecifier> "
      "<feature-wspecifier>\n"
      "e.g. apply-cmvn-online-cuda 'matrix-sum scp:data/train/cmvn.scp -|' data/train/split8/1/feats.scp ark:-\n";

    ParseOptions po(usage);

    OnlineCmvnOptions cmvn_opts;

    std::string spk2utt_rspecifier;
    cmvn_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    std::string global_stats_rxfilename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      feature_wspecifier = po.GetArg(3);

    // global_cmvn_stats helps us initialize to online CMVN to
    // reasonable values at the beginning of the utterance.
    Matrix<double> global_cmvn_stats;
    ReadKaldiObject(global_stats_rxfilename, &global_cmvn_stats);

    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    int32 num_done = 0;
    int64 tot_t = 0;
      
    OnlineCmvnState cmvn_state(global_cmvn_stats);
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);
    CudaOnlineCmvn cuda_cmvn(cmvn_opts, cu_cmvn_state);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      int32_t numRows = feats.NumRows();
      int32_t numCols = feats.NumCols();

      CuMatrix<BaseFloat> cu_feats_in(feats);
      CuMatrix<BaseFloat> cu_feats_out(numRows, numCols, kUndefined);
      Matrix<BaseFloat> normalized_feats(numRows, numCols, kUndefined);

      cuda_cmvn.ComputeFeatures(cu_feats_in, &cu_feats_out);

      normalized_feats.CopyFromMat(cu_feats_out);

      num_done++;
      tot_t += feats.NumRows();
      feature_writer.Write(utt, normalized_feats);

      num_done++;
    }

    KALDI_LOG << "Applied online CMVN to " << num_done << " files, or "
      << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

