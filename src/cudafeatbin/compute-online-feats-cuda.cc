// cudafeatbin/compute-online-feats-cuda.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if HAVE_CUDA == 1
#include <nvToolsExt.h>
#endif
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudafeat/online-cuda-feature-pipeline.h"
#include "feat/wave-reader.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Extract features and ivectors for utterances using the cuda online\n"
      "feature pipeline. This class models the online feature pipeline.\n"  
      "\n"
      "Usage:  compute-online-feats-cuda [options] <wave-rspecifier> "
      "<ivector-wspecifier> <feats-wspecifier>\n"
      "e.g.: \n"
      "  ./compute-online-feats-cuda --config=feature_config wav.scp "
      "ark,scp:ivector.ark,ivector.scp ark,scp:feat.ark,feat.scp\n";

    ParseOptions po(usage);
    // Use online feature config as that is the flow we are trying to model
    OnlineNnet2FeaturePipelineConfig feature_opts;

    feature_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    std::string wav_rspecifier = po.GetArg(1),
      ivector_wspecifier = po.GetArg(2),
      feature_wspecifier = po.GetArg(3);

    OnlineCudaFeaturePipeline feature_pipeline(feature_opts);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatVectorWriter ivector_writer;
    BaseFloatMatrixWriter feature_writer;

    if (!ivector_writer.Open(ivector_wspecifier)) {
      KALDI_ERR << "Could not initialize ivector_writer with wspecifier "
        << ivector_wspecifier;
    }
    if (!feature_writer.Open(feature_wspecifier)) {
      KALDI_ERR << "Could not initialize feature_writer with wspecifier "
        << feature_wspecifier;
    }

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      KALDI_LOG << "Processing Utterance " << utt;
      try
      {
        const WaveData &wave_data = reader.Value();
        SubVector<BaseFloat> waveform(wave_data.Data(), 0);
        CuVector<BaseFloat> cu_wave(waveform);
        CuMatrix<BaseFloat> cu_features;
        CuVector<BaseFloat> cu_ivector;

        nvtxRangePushA("Feature Extract");
        feature_pipeline.ComputeFeatures(cu_wave,  wave_data.SampFreq(),
            &cu_features, &cu_ivector);
        cudaDeviceSynchronize();
        nvtxRangePop();

        Matrix<BaseFloat> features(cu_features.NumRows(), cu_features.NumCols());
        Vector<BaseFloat> ivector(cu_ivector.Dim());

        features.CopyFromMat(cu_features);
        ivector.CopyFromVec(cu_ivector);

        feature_writer.Write(utt, features);
        ivector_writer.Write(utt, ivector);

        num_success++;
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
          << utt;
        continue;
      }
    }
    KALDI_LOG << "Processed " << num_utts << " utterances with "
      << num_utts - num_success << " failures.";
    return (num_success != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}
