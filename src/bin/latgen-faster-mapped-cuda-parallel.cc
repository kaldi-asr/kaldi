// bin/latgen-faster-mapped.cc

// Copyright      2018  Zhehuai Chen

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


#include <nvToolsExt.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"

#include "decoder/cuda-decoder-utils.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/lattice-faster-decoder-cuda.h"
#include "decoder/cuda-lattice-decoder.h"
#include <omp.h>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
      "Generate lattices, reading log-likelihoods as matrices\n"
      " (model is needed only for the integer mappings in its transition-model)\n"
      "Usage: latgen-faster-mapped [options] trans-model-in (fst-in|fsts-rspecifier) loglikes-rspecifier"
      " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    CudaLatticeDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    config.gpu_fraction = 1.0 / omp_get_max_threads();
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
                fst_in_str = po.GetArg(2),
                feature_rspecifier = po.GetArg(3),
                lattice_wspecifier = po.GetArg(4),
                words_wspecifier = po.GetOptArg(5),
                alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    double elapsed = 0;
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      Fst<StdArc> *decode_fst;
      decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      cuInit(0);
      CudaFst decode_fst_cuda;
      #pragma omp parallel shared(po, decode_fst_cuda)
      {
        printf("Thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
#if HAVE_CUDA==1
        CuDevice::Instantiate().SelectGpuId("yes");
        CuDevice::Instantiate().AllowMultithreading();
#endif
        /*
        #if 1
              cuInit(0);
              cudaDeviceReset();
              //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
              //cudaSetDeviceFlags(cudaDeviceScheduleYield);
              //cudaSetDeviceFlags(cudaDeviceScheduleSpin);
              uint flags;
              cudaGetDeviceFlags(&flags);
              KALDI_VLOG(3)<<flags;
              //assert(flags&cudaDeviceScheduleBlockingSync);
        #else
              CuDevice::Instantiate().SelectGpuId("yes");
              CuDevice::Instantiate().AllowMultithreading();
        #endif
        */
        #pragma omp barrier
        SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
        // Input FST is just one FST, not a table of FSTs.

        if (omp_get_thread_num() == 0) {
          decode_fst_cuda.initialize(*decode_fst);
        }
        #pragma omp barrier

        LatticeFasterDecoderCuda decoder(decode_fst_cuda, config);
        {

          for (; !loglike_reader.Done(); loglike_reader.Next()) {
            if (omp_get_thread_num() == 0) {
              printf("cudaMallocMemory: %lg GB, cudaMallocManagedMemory: %lg GB\n",
                     (decoder.Decoder().GetCudaMallocBytes()*omp_get_num_threads() +
                      decode_fst_cuda.GetCudaMallocBytes()) / 1024.0 / 1024 / 1024,
                     decoder.Decoder().GetCudaMallocManagedBytes() / 1024.0 / 1024 / 1024 *
                     omp_get_num_threads());
            }
            nvtxRangePushA("whole decoding");
            nvtxRangePushA("before_decoding");
            if (omp_get_thread_num() == 0) timer.Reset();
            #pragma omp barrier

            std::string utt = loglike_reader.Key();
            Matrix<BaseFloat> loglikes (loglike_reader.Value());
            loglike_reader.FreeCurrent();
            if (loglikes.NumRows() == 0) {
              KALDI_WARN << "Zero-length utterance: " << utt;
              num_fail++;
              continue;
            }

            DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

            nvtxRangePop();

            double like;
            Lattice lat;
            if (DecodeUtteranceLatticeFasterCuda(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like,
                  &lat)) {
              tot_like += like;
              frame_count += loglikes.NumRows();
              num_success++;
            } else num_fail++;
            #pragma omp barrier
            if (omp_get_thread_num() == 0) elapsed += timer.Elapsed();
            nvtxRangePop();
            #pragma omp critical
            {
              DecodeUtteranceLatticeFasterCudaOutput(
                decoder, decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like,
                lat);
            }
          }
        }
        #pragma omp barrier
      }//omp parallel
      delete decode_fst; // delete this only after decoder goes out of scope.
      decode_fst_cuda.finalize();
      cudaDeviceSynchronize();
    } else { // We have different FSTs for different utterances.
      KALDI_ERR << "unfinished";
    }

    KALDI_LOG << "Time taken " << elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like / frame_count) <<
              " over "
              << frame_count << " frames.";
    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
