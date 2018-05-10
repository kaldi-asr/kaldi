// This is the main program of Radit's Final Project
// The file mostly contains the same program with onlinebin/online-wav-gmm-decode-faster.cpp

// Uses Parallel Viterbi Beam Search Algorithm from Arturo Argueta and David Chiang's paper.

#include "gpu/gpu-online-decodable.h"
#include "gpu/gpu-diag-gmm.h"
#include "gpufst/fst.h"
#include "gpufst/gpu-fst.h"
#include "gpufst/prob-ptr.h"
#include "gpufst/numberizer.h"

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
        << std::endl; cudaDeviceSynchronize(); } while (0)

int ceildiv(int x, int y) { return (x-1)/y+1; }
#define BLOCK_SIZE 512
#define BEAM_SIZE 10
#define BATCH_SIZE 27

using namespace kaldi;

__device__ void cobaKernelGPUDiagGmm(GPUDiagGmm *G){ 
  printf("DEVICE G->valid_gconsts_ : %d\n", G->valid_gconsts_);
  printf("DEVICE G->gconsts_:\n");
  for(int i = 0;i < G->gconsts_.Dim(); ++i){
    printf("DEVICE G->gconsts[%d] : %.10f\n", i, G->gconsts_.data[i]);
  }
}

__global__ void cobaKernelGPUDiagGmmHost(GPUDiagGmm *G){
  cobaKernelGPUDiagGmm(G);
}

void cobaGPUDiagGmm(GPUDiagGmm *G){
  fprintf(stderr, "HOST G->valid_gconsts_ : %d\n", G->valid_gconsts_);
  fprintf(stderr, "HOST G->gconsts_:\n");
//  for(int i = 0;i < G->gconsts_.Dim(); ++i){
//    fprintf(stderr, "HOST G->gconsts[%d] : %.f\n", i, G->gconsts_.data[i]);
//  }
}

__global__ void AddPdfToGPUAmGmm(GPUAmDiagGmm* gpu_am_gmm, int pdf_idx, GPUDiagGmm *gpu_gmm){
  gpu_am_gmm->densities[pdf_idx] = gpu_gmm;
}

__device__ void cobaKernelGPUAmDiagGmm(GPUAmDiagGmm *G){
  cobaKernelGPUDiagGmm(G->densities[0]);
}

__global__ void cobaKernelGPUAmDiagGmmHost(GPUAmDiagGmm *G){
  cobaKernelGPUAmDiagGmm(G);
}

__global__ void tesAkhir(GPUOnlineDecodableDiagGmmScaled* gpu_decodable){
  printf("TES AKHIR\n");
  printf("Fase 1 : GPUAmDiagGmm\n");
  GPUAmDiagGmm* gpu_am_gmm = gpu_decodable->ac_model_;
  GPUDiagGmm **densities = gpu_am_gmm->densities;
  GPUDiagGmm* density = densities[0]; 
  cobaKernelGPUDiagGmm(density);
  printf("Fase 2 : GPUTransitionModel\n"); 
  printf("Fase 3 : AC Model\n");
  printf("acoustic scale : %.10f\n", gpu_decodable->ac_scale_);
}

int main(int argc, char *argv[]) {
  try {

    typedef kaldi::int32 int32;
    typedef OnlineFeInput<Mfcc> FeInput;

    // up to delta-delta derivative features are calculated (unless LDA is used)
    const int32 kDeltaOrder = 2;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding.\n"
        "Writes integerized-text and .ali files for WER computation. Utterance "
        "segmentation is done on-the-fly.\n"
        "Feature splicing/LDA transform is used, if the optional(last) argument "
        "is given.\n"
        "Otherwise delta/delta-delta(i.e. 2-nd order) features are produced.\n"
        "Caution: the last few frames of the wav file may not be decoded properly.\n"
        "Hence, don't use one wav file per utterance, but "
        "rather use one wav file per show.\n\n"
        "Usage: online-wav-gmm-decode-faster [options] wav-rspecifier model-in"
        "fst-in word-symbol-table silence-phones transcript-wspecifier "
        "alignments-wspecifier [lda-matrix-in]\n\n"
        "Example: ./online-wav-gmm-decode-faster --rt-min=0.3 --rt-max=0.5 "
        "--max-active=4000 --beam=12.0 --acoustic-scale=0.0769 "
        "scp:wav.scp model HCLG.fst words.txt '1:2:3:4:5' ark,t:trans.txt ark,t:ali.txt";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    int32 cmn_window = 600,
      min_cmn_window = 100; // adds 1 second latency, only at utterance start.
    int32 channel = -1;
    int32 right_context = 4, left_context = 4;

    OnlineFasterDecoderOpts decoder_opts;
    decoder_opts.Register(&po, true);
    OnlineFeatureMatrixOptions feature_reading_opts;
    feature_reading_opts.Register(&po);
    
    po.Register("left-context", &left_context, "Number of frames of left context");
    po.Register("right-context", &right_context, "Number of frames of right context");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("cmn-window", &cmn_window,
        "Number of feat. vectors used in the running average CMN calculation");
    po.Register("min-cmn-window", &min_cmn_window,
                "Minumum CMN window used at start of decoding (adds "
                "latency only at start)");
    po.Register("channel", &channel,
        "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Read(argc, argv);

    if (po.NumArgs() != 7 && po.NumArgs() != 8) {
      po.PrintUsage();
      return 1;
    }
    
    std::string wav_rspecifier = po.GetArg(1),
        model_rspecifier = po.GetArg(2),
        fst_rspecifier = po.GetArg(3),
        word_syms_filename = po.GetArg(4),
        silence_phones_str = po.GetArg(5),
        words_wspecifier = po.GetArg(6),
        alignment_wspecifier = po.GetArg(7),
        lda_mat_rspecifier = po.GetOptArg(8);

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
        KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
        KALDI_ERR << "No silence phones given!";

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    Matrix<BaseFloat> lda_transform;
    if (lda_mat_rspecifier != "") {
      bool binary_in;
      Input ki(lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
    }

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
        bool binary;
        Input ki(model_rspecifier, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_gmm.Read(ki.Stream(), binary);
    }
    
    gpufst::gpu_fst m = gpufst::read_fst_noNumberizer(fst_rspecifier); // harus berupa file text
    
    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
    int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

    int32 window_size = right_context + left_context + 1;
    decoder_opts.batch_size = std::max(decoder_opts.batch_size, window_size);
    
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    for (; !reader.Done(); reader.Next()) {
      std::string wav_key = reader.Key();
      std::cerr << "File: " << wav_key << std::endl;
      const WaveData &wav_data = reader.Value();
      if(wav_data.SampFreq() != 16000)
        KALDI_ERR << "Sampling rates other than 16kHz are not supported!";
      int32 num_chan = wav_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << wav_key << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }
      OnlineVectorSource au_src(wav_data.Data().Row(this_chan));
      Mfcc mfcc(mfcc_opts);
      FeInput fe_input(&au_src, &mfcc,
                       frame_length*(wav_data.SampFreq()/1000),
                       frame_shift*(wav_data.SampFreq()/1000));
      OnlineCmnInput cmn_input(&fe_input, cmn_window, min_cmn_window);
      OnlineFeatInputItf *feat_transform = 0;
      if (lda_mat_rspecifier != "") {
        feat_transform = new OnlineLdaInput(
            &cmn_input, lda_transform,
            left_context, right_context);
      } else {
        DeltaFeaturesOptions opts;
        opts.order = kDeltaOrder;
        feat_transform = new OnlineDeltaInput(opts, &cmn_input);
      }

      // feature_reading_opts contains number of retries, batch size.
      OnlineFeatureMatrix feature_matrix(feature_reading_opts,
                                         feat_transform);
      
      OnlineDecodableDiagGmmScaled decodable(am_gmm, trans_model, acoustic_scale,
                                             &feature_matrix);
      
      // Copy AmDiagGmm ke GPUAmDiagGmm
      GPUAmDiagGmm gpu_am_gmm_h;
      GPUAmDiagGmm *gpu_am_gmm_d;
      gpu_am_gmm_h.densities_.resize(am_gmm.NumPdfs());
      gpu_am_gmm_h.densities = gpu_am_gmm_h.densities_.data().get();

      cudaMalloc((void**) &gpu_am_gmm_d, sizeof(GPUAmDiagGmm));
      cudaMemcpy(gpu_am_gmm_d, &gpu_am_gmm_h, sizeof(GPUAmDiagGmm), cudaMemcpyHostToDevice);

      for(size_t i = 0; i < am_gmm.NumPdfs(); ++i){
        GPUDiagGmm gpu_gmm_h(am_gmm.GetPdf(i));
        if(i == 0){
          const Vector<BaseFloat>& gconstsh = am_gmm.GetPdf(i).gconsts();
          const BaseFloat* gconstsh_data = gconstsh.Data();
          for(int j = 0;j < gconstsh.Dim(); ++j){
            printf("HOST gconstsh[%d] : %.10f\n", j, gconstsh_data[j]);
          }
        }

        GPUDiagGmm *gpu_gmm_d;
        cudaMalloc((void**) &gpu_gmm_d, sizeof(GPUDiagGmm));
        cudaMemcpy(gpu_gmm_d, &gpu_gmm_h, sizeof(GPUDiagGmm), cudaMemcpyHostToDevice);
        if(i == 0){
          cobaKernelGPUDiagGmmHost<<<1,1>>>(gpu_gmm_d);
        }
        AddPdfToGPUAmGmm<<<1,1>>>(gpu_am_gmm_d, i, gpu_gmm_d);
      }
      // Copy TransitionModel ke GPUTransitionModel
      cobaKernelGPUAmDiagGmmHost<<<1,1>>>(gpu_am_gmm_d);
      
      GPUTransitionModel gpu_trans_model_h(trans_model);
      GPUTransitionModel *gpu_trans_model_d;
      cudaMalloc((void**) &gpu_trans_model_d, sizeof(GPUTransitionModel));
      cudaMemcpy(gpu_trans_model_d, &gpu_trans_model_h, sizeof(GPUTransitionModel), cudaMemcpyHostToDevice); 
      // Create GPUOnlineDecodableDiagGmmScaled
      GPUOnlineDecodableDiagGmmScaled gpu_decodable_h(gpu_am_gmm_d, gpu_trans_model_d, acoustic_scale);
      GPUOnlineDecodableDiagGmmScaled* gpu_decodable_d;
      cudaMalloc((void**) &gpu_decodable_d, sizeof(GPUOnlineDecodableDiagGmmScaled));
      cudaMemcpy(gpu_decodable_d, &gpu_decodable_h, sizeof(GPUOnlineDecodableDiagGmmScaled), cudaMemcpyHostToDevice);

      // TODO : BARU MASUKIN VITERBINYA 
      tesAkhir<<<1,1>>>(gpu_decodable_d);
      cudaFree(gpu_decodable_d);
      cudaFree(gpu_trans_model_d);
      for(size_t i = 0;i < am_gmm.NumPdfs(); ++i){
        cudaFree(gpu_am_gmm_h.densities_[i]);
      }
      cudaFree(gpu_am_gmm_d);
      
      delete feat_transform;
    }

    std::cerr << "SAMPE DI AKHIR BANGET" << std::endl;

    return 0;
  } catch(const std::exception& e) { 
    std::cerr << "MASUK CATCH" << std::endl;
    std::cerr << e.what();
    return -1;
  }
} // main()
