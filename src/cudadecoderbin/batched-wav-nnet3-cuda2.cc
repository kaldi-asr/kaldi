// cudadecoderbin/batched-wav-nnet3-cuda2.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#include <atomic>
#if HAVE_CUDA == 1

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <sstream>

#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"
#include "cudadecoderbin/cuda-bin-tools.h"
#include "cudamatrix/cu-allocator.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

// Used as a max segment length if segmentation is disabled
// Not using FLT_MAX to avoid overflows
#define KALDI_CUDA_DECODER_BIN_MAX_SEGMENT_LENGTH_S 3600

using namespace kaldi;
using namespace cuda_decoder;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and decodes them with "
        "neural nets\n"
        "(nnet3 setup).  Note: some configuration values "
        "and inputs "
        "are\n"
        "set via config files whose filenames are passed as "
        "options\n"
        "Output can either be a lattice wspecifier or a ctm filename"
        "\n"
        "Usage: batched-wav-nnet3-cuda2 [options] <nnet3-in> "
        "<fst-in> "
        "<wav-rspecifier> <lattice-wspecifier|ctm-wxfilename>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int iterations = 1;
    bool segmentation = false;
    std::string lattice_postprocessor_config_rxfilename;
    ParseOptions po(usage);
    po.Register("write-lattice", &write_lattice,
                "Output lattice to a file. Setting to false is useful when "
                "benchmarking");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo,
                "Limits the number of files that are processed by "
                "this driver. "
                "After N files are processed the remaining files "
                "are ignored. "
                "Useful for profiling");
    po.Register("iterations", &iterations,
                "Number of times to decode the corpus. Output will "
                "be written "
                "only once.");
    po.Register("segmentation", &segmentation,
                "Split audio files into segments");
    po.Register("lattice-postprocessor-rxfilename",
                &lattice_postprocessor_config_rxfilename,
                "(optional) Config file for lattice postprocessor");

    // Multi-threaded CPU and batched GPU decoder
    BatchedThreadedNnet3CudaPipeline2Config batched_decoder_config;
    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);
    batched_decoder_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    std::string nnet3_rxfilename = po.GetArg(1), fst_rxfilename = po.GetArg(2),
                wav_rspecifier = po.GetArg(3), output_wspecifier = po.GetArg(4);
    std::shared_ptr<TransitionModel> trans_model(new TransitionModel());

    nnet3::AmNnetSimple am_nnet;

    // read transition model and nnet
    bool binary;
    Input ki(nnet3_rxfilename, &binary);
    trans_model->Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));

    std::unique_ptr<CompactLatticeWriter> clat_writer;
    std::unique_ptr<Output> ctm_writer;
    OpenOutputHandles(output_wspecifier, &clat_writer, &ctm_writer);
    if (!write_lattice) clat_writer.reset();
    std::mutex output_writer_m_;

    fst::Fst<fst::StdArc> *decode_fst =
        fst::ReadFstKaldiGeneric(fst_rxfilename);

    if (!segmentation) {
      batched_decoder_config.seg_opts.segment_length_s =
          KALDI_CUDA_DECODER_BIN_MAX_SEGMENT_LENGTH_S;
    }
    BatchedThreadedNnet3CudaPipeline2 cuda_pipeline(
        batched_decoder_config, *decode_fst, am_nnet, *trans_model);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "") {
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
      cuda_pipeline.SetSymbolTable(*word_syms);
    }

    // Lattice postprocessor
    if (lattice_postprocessor_config_rxfilename.empty()) {
      if (ctm_writer) {
        KALDI_ERR << "You must configure the lattice postprocessor with "
                     "--lattice-postprocessor-rxfilename to use CTM output";
      }
    } else {
      LoadAndSetLatticePostprocessor(lattice_postprocessor_config_rxfilename,
                                     &cuda_pipeline);
    }

    int32 num_task_submitted = 0, num_err = 0;
    double total_audio = 0;

    nvtxRangePush("Global Timer");
    // starting timer here so we
    // can measure throughput
    // without allocation
    // overheads
    // using kaldi timer, which starts counting in the constructor
    Timer timer;
    std::vector<double> iteration_timer;
    KALDI_LOG << "Inferencing...";
    for (int iter = 0; iter < iterations; iter++) {
      num_task_submitted = 0;
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        std::string key = utt;
        if (iter > 0) key = std::to_string(iter) + "-" + key;
        std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
        wave_data->Swap(&wav_reader.Value());
        if (iter == 0) {
          // calculating number of utterances per
          // iteration calculating total audio
          // time per iteration
          total_audio += wave_data->Duration();
        }

        // Callback used when results are ready
        //
        // If lattice output, write all lattices to clat_writer
        // If segmentation is true, then the keys are:
        // [utt_key]-[segment_offset]
        //
        // If CTM output, merging segment results together
        // and writing this single output to ctm_writer
        SegmentedResultsCallback segmented_callback =
            [&clat_writer, &ctm_writer, &output_writer_m_, key, segmentation,
             word_syms](SegmentedLatticeCallbackParams &params) {
              if (clat_writer) {
                std::lock_guard<std::mutex> lk(output_writer_m_);
                bool print_offsets = segmentation;
                WriteLattices(params.results, key, print_offsets, *clat_writer);
              }

              if (ctm_writer) {
                std::lock_guard<std::mutex> lk(output_writer_m_);
                MergeSegmentsToCTMOutput(params.results, key,
                                         ctm_writer->Stream(), word_syms);
              }
            };

        int result_type = 0;
        if (ctm_writer) result_type |= CudaPipelineResult::RESULT_TYPE_CTM;
        if (clat_writer) result_type |= CudaPipelineResult::RESULT_TYPE_LATTICE;

        // Always calling SegmentedResultsCallback even if segmentation is false
        // If segmentation is false, we just set segment_length to some high
        // value. This is to avoid unnecessary code duplication
        cuda_pipeline.SegmentedDecodeWithCallback(wave_data, segmented_callback,
                                                  result_type);
        num_task_submitted++;
        if (num_todo != -1 && num_task_submitted >= num_todo) break;
      }  // end utterance loop
    }    // end iterations loop

    cuda_pipeline.WaitForAllTasks();

    // number of seconds elapsed since the creation of timer
    double total_time = timer.Elapsed();
    nvtxRangePop();

    KALDI_LOG << "Decoded " << num_task_submitted << " utterances, " << num_err
              << " with errors.";

    KALDI_LOG << "Overall: "
              << " Aggregate Total Time: " << total_time
              << " Total Audio: " << total_audio * iterations
              << " RealTimeX: " << total_audio * iterations / total_time;

    delete word_syms;  // will delete if non-NULL.

    cudaDeviceSynchronize();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()

#endif  // if HAVE_CUDA == 1
