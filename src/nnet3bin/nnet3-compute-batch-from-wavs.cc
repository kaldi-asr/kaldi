// nnet3bin/nnet3-compute-batch-from-wavs.cc

// Copyright 2012-2018   Johns Hopkins University (author: Daniel Povey)
//           2020-2021   Xiaomi Corporation (Author: Zhao Yan)

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
#include "feat/wave-reader.h"
#include "matrix/kaldi-vector.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/decodable-batch-looped.h"
#include "online2/online-nnet2-feature-pipeline.h"

using namespace kaldi;
using namespace kaldi::nnet3;

typedef struct _ThreadPara {
  std::vector<std::string> wavs_file_name;
  std::vector<std::string> utts;
  BaseFloatMatrixWriter    *matrix_writer;
  std::mutex               mtx;
  int32                    index;
  
  OnlineNnet2FeaturePipelineInfo *pipeline_info;
  NnetBatchLoopedComputer        *computer;
  TransitionModel                *trans_model;
} ThreadPara; 


void *ThreadFunction(void *para) {
  ThreadPara *thread_para = reinterpret_cast<ThreadPara *>(para);
  int32 output_dim = thread_para->computer->GetInfo().output_dim;
  const int kPackageSize = 512;
  
  while (true) {
    std::string utt, wav_file_name;
    
    {
      std::unique_lock<std::mutex> lck(thread_para->mtx);
      if (thread_para->index >= thread_para->wavs_file_name.size())
        break;
      
      utt = thread_para->utts[thread_para->index];
      wav_file_name = thread_para->wavs_file_name[thread_para->index];
      thread_para->index++;
    }

    OnlineNnet2FeaturePipeline feature_pipeline(*(thread_para->pipeline_info));
    DecodableNnetBatchLoopedOnline decodable(
        thread_para->computer, *(thread_para->trans_model), 
        feature_pipeline.InputFeature(), 
        feature_pipeline.IvectorFeature());
    std::vector< Vector<BaseFloat> > likelihoods;
    
    bool binary = true;
    Input ki(wav_file_name, &binary);
    WaveHolder wave_holder;
    if (!wave_holder.Read(ki.Stream())) 
      KALDI_ERR << "Read failure from " << wav_file_name;
    
    const Matrix<BaseFloat> &waveform = wave_holder.Value().Data();
    int32 num_samplers_read = 0, num_frames_read = 0;
    BaseFloat samp_freq = wave_holder.Value().SampFreq();

    while (num_samplers_read < waveform.NumCols()) {
      Vector<BaseFloat> wave_data;
      int32 num_samplers = num_samplers_read + kPackageSize <= waveform.NumCols()
                           ? kPackageSize
                           : (waveform.NumCols() - num_samplers_read);
      
      wave_data.CopyColsFromMat(waveform.ColRange(num_samplers_read, num_samplers));
      feature_pipeline.AcceptWaveform(samp_freq, wave_data);
      if (decodable.NumFramesReady() > num_frames_read) {
        likelihoods.resize(decodable.NumFramesReady(), Vector<BaseFloat>(output_dim));
        while (num_frames_read < decodable.NumFramesReady()) {
          decodable.GetOutputForFrame(num_frames_read, &(likelihoods[num_frames_read]));
          num_frames_read++;
        }
      }

      num_samplers_read += num_samplers;
    }

    if (num_frames_read > 0) {
      std::unique_lock<std::mutex> lck(thread_para->mtx);
      Matrix<BaseFloat> matrix(num_frames_read, likelihoods[0].Dim());
      for (int32 i = 0; i < num_frames_read; i++) 
        matrix.CopyRowFromVec(likelihoods[i], i);
      thread_para->matrix_writer->Write(utt, matrix);
    }
    else {
      KALDI_WARN << "Propagate failed for " << utt;
    }
  }

  return NULL;
}


int main(int argc, char *argv[]) {
  try {
#if HAVE_CUDA==1
    const char *usage =
        "Propagate the features through raw neural network model. "
        "This version is running on GPU only. "
        "\n"
        "Usage: nnet3-compute-batch-from-wavs [options] <trans_model> <wavs-rspecifier> <matrixs-wspecifier>\n"
        " e.g.: nnet3-compute-batch-from-wavs final.mdl ark:wavs.scp ark:out.ark\n";

    ParseOptions po(usage);
    Timer timer;

    NnetBatchLoopedComputationOptions computation_config;
    OnlineNnet2FeaturePipelineConfig pipeline_config;

    CuDevice::RegisterDeviceOptions(&po);
    pipeline_config.Register(&po);
    computation_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    CuDevice::Instantiate().AllowMultithreading();
    CuDevice::Instantiate().SelectGpuId("yes");

    std::string trans_model_rxfilename = po.GetArg(1),
                wav_rspecifier = po.GetArg(2),
                matrix_wspecifier = po.GetArg(3);

    TransitionModel trans_model;
    Nnet raw_nnet;
    AmNnetSimple am_nnet;
    bool binary;
    Input ki(trans_model_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);

    Nnet &nnet = am_nnet.GetNnet();
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    DecodableNnetBatchLoopedInfo decodable_info(computation_config, &nnet);
    NnetBatchLoopedComputer computer(decodable_info);
    OnlineNnet2FeaturePipelineInfo pipeline_info(pipeline_config);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    
    ThreadPara thread_para;
    thread_para.index = 0;
    thread_para.wavs_file_name.clear();
    thread_para.matrix_writer = &matrix_writer;
    thread_para.pipeline_info = &pipeline_info;
    thread_para.computer = &computer;
    thread_para.trans_model = &trans_model;

    SequentialTokenReader wav_reader(wav_rspecifier);
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string utt_id = wav_reader.Key();
      const std::string &wav_file_name = wav_reader.Value();
      thread_para.utts.push_back(utt_id);
      thread_para.wavs_file_name.push_back(wav_file_name);
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) 
      threads.push_back(std::thread(ThreadFunction, &thread_para));

    for (std::size_t i = 0; i < threads.size(); i++) 
      threads[i].join();

    CuDevice::Instantiate().PrintProfile();
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed << "s";
    return 0;
#else
    KALDI_ERR << "This program requires CUDA available.";
#endif
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
