// This is the main program of Radit's Final Project
// The file mostly contains the same program with onlinebin/online-wav-gmm-decode-faster.cpp

// Uses Parallel Viterbi Beam Search Algorithm from Arturo Argueta and David Chiang's paper.

#include "gpu/gpu-online-decodable.h"
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

int ceildiv(int x, int y) { return (x-1)/y+1; }
#define BLOCK_SIZE 512
#define BEAM_SIZE 10
#define BATCH_SIZE 27


#define EPS_SYM 0

using namespace gpufst;

int frame_;
int utt_frames_;


struct PointerComparison
{
    __host__ __device__ bool operator()(const prob_ptr_t &x, const prob_ptr_t &y)
    {
        float prob_x = unpack_prob(x);
        float prob_y = unpack_prob(y);
        return prob_x > prob_y;
    }
}; 

__global__ void compute_initial(int *from_states, int *to_states, float *probs,
        int start_offset, int end_offset,
        state_t initial_state,
        prob_ptr_t *viterbi)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int offset = start_offset+id;

  __shared__ int from_shared_states[BLOCK_SIZE];
  __shared__ int to_shared_states[BLOCK_SIZE];
  __shared__ float shared_probs[BLOCK_SIZE];
  __shared__ sym_t shared_inputs[BLOCK_SIZE];

  from_shared_states[threadIdx.x] = from_states[offset];
  to_shared_states[threadIdx.x] = to_states[offset];
  shared_probs[threadIdx.x] = probs[offset];
  shared_inputs[threadIdx.x] = inputs[offset];

  if (offset < end_offset && from_shared_states[threadIdx.x] == initial_state && shared_inputs[threadIdx.x] == EPS_SYM) {
    atomicMax(&viterbi[to_shared_states[threadIdx.x]], pack(shared_probs[threadIdx.x], offset));
  }
}

/* TODO:
 * 1. Ganti assignment pp di if itu (tambahin LogLikelihood)
 *    Perhatiin LogLikelihood bakal assign sesuatu di CUDA, jangan sampe kegedean
 */
__global__ void compute_transition(int *from_states, int *to_states, float *probs,
          int start_offset, int end_offset, int frame,
          prob_ptr_t *viterbi_prev, prob_ptr_t *viterbi) {
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int offset = start_offset+id;

  __shared__ int from_shared_states[BLOCK_SIZE];
  __shared__ int to_shared_states[BLOCK_SIZE];
  __shared__ prob_ptr_t viterbi_from_shared_states[BLOCK_SIZE];
  __shared__ float shared_probs[BLOCK_SIZE];
  

  from_shared_states[threadIdx.x] = from_states[offset];
  to_shared_states[threadIdx.x] = to_states[offset];
  shared_probs[threadIdx.x] = probs[offset];
  viterbi_from_shared_states[threadIdx.x] = viterbi_prev[from_shared_states[threadIdx.x]];

  if (offset < end_offset && from_shared_states[threadIdx.x] != EPS_SYM) {
    // TODO : GANTI assignment pp dibawah ini
    prob_ptr_t pp = pack(unpack_prob(viterbi_from_shared_states[threadIdx.x]) + shared_probs[threadIdx.x], offset);
    atomicMax(&viterbi[/*to_register_state*/to_shared_states[threadIdx.x]], pp);
  }
}

__global__ void compute_final(int *final_states, float *final_probs,
            int num_finals,
            prob_ptr_t *viterbi_prev, prob_ptr_t *viterbi) {
  // To do: this should be a reduction instead
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < num_finals) {
    prob_ptr_t pp = pack(unpack_prob(viterbi_prev[final_states[id]]) + final_probs[id], final_states[id]);
    atomicMax(viterbi, pp);
  }
}

__global__ void get_path(int *from_nodes,
            prob_ptr_t *viterbi,
            int batch_length, int num_nodes,
            prob_ptr_t *path) {
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id == 0) {
    int state = unpack_ptr(path[batch_length]);
    for (int t = batch_length - 1; t >= 0; t--) {
      prob_ptr_t pp = viterbi[t * num_nodes+state];
      path[t] = pp;
      state = from_nodes[unpack_ptr(pp)];
    }
  }
}

/* TODO
 * 1. Ganti Input Symbols jadi decodablenya
 * 2. Ganti Resizenya jadi Batch Size
 * 3. Konsep Beam Search disini beda, disini prune banyak state (sama dengan --max-active-state), di kaldi max prob
 * 4. (DONE) Input Offsets kayaknya gaperlu, semuanya dipake sekarang soalnya
 * 5. Kasih offset frame di fungsi utama
 * 6. Compute Final ga harus dari final stat e. kalo misalnya dia ga nyampe final state maka cari yang terbaik.
 * 7. (DONE) Ganti yang dapet output symbol dari input symbol. SIZE ga harus sama.
 * 8.  Abis batch frame, itu ga harus dari m.initial, ini harus coba dicari lagi mulainya dari mana.
 */
prob_t viterbi(gpu_fst &m, OnlineDecodableDiagGmmScaled* decodable, GPUOnlineDecodableDiagGmmScaled* gpu_decodable, vector<sym_t> &output_symbols) {
  int verbose=0;

  int batch_frame = 0;

  static thrust::device_vector<prob_ptr_t> viterbi;
  viterbi.resize((BATCH_SIZE + 1) * m.num_states); // TODO : instead of input symbols, kasih batch_size
  prob_ptr_t init_value = pack(-FLT_MAX, 0);
  thrust::fill(viterbi.begin(), viterbi.end(), init_value);

  thrust::device_vector<prob_ptr_t> path(BATCH_SIZE + 1);
  int start_offset = 0; 
  int end_offset = m.input_offsets.back();

  compute_initial <<<ceildiv(end_offset-start_offset, BLOCK_SIZE), BLOCK_SIZE>>> (
    m.from_states.data().get(),
    m.to_states.data().get(),
    m.probs.data().get(),
    start_offset, end_offset,
    m.initial,
    viterbi.data().get()
  );

  if (verbose) {
    for (auto pp: viterbi)
      std::cout << unpack_prob(pp) << " ";
    std::cout << std::endl;
  }

  for (; !decodable->IsLastFrame(frame_ - 1) && batch_frame < BATCH_SIZE;
     ++frame_, ++utt_frames_, ++batch_frame) {

    // DO SOMETHING HERE : update cur_feats
    gpu_decodable.cur_feats_ = GPUVector<BaseFloat>(decodable->cur_feats());

    compute_transition <<<ceildiv(end_offset-start_offset, BLOCK_SIZE), BLOCK_SIZE>>> (
      m.from_states.data().get(), 
      m.to_states.data().get(),
      m.probs.data().get(),
      start_offset, end_offset, 
      frame_,
      viterbi.data().get() + t*m.num_states,
      viterbi.data().get() + (t+1)*m.num_states);
    
    if (verbose) {
      for (auto pp: viterbi)
        std::cout << unpack_prob(pp) << " ";
      std::cout << std::endl;
    }
  }

  // TODO : cari token yang terbaik terlebih dahulu instead of final token
  compute_final <<<ceildiv(m.final_states.size(), 1024), 1024>>> (
    m.final_states.data().get(),
    m.final_probs.data().get(),
    m.final_states.size(),
    viterbi.data().get() + batch_frame * m.num_states,
    (&path.back()).get());

  get_path <<<1,1>>> (
    m.from_states.data().get(),
    viterbi.data().get(), 
    batch_frame + 1, m.num_states,
    path.data().get()
  );

  cudaError_t e = cudaGetLastError();                                 
  if (e != cudaSuccess) {                                              
    std::cerr << "CUDA failure: " << cudaGetErrorString(e) << std::endl;
    exit(1);
  }

  thrust::host_vector<prob_ptr_t> h_path(path);
  output_symbols.clear();

  // TODO : ini ga harus dapet dari sini, pasti lebih dikit soalnya (jumlah kata << jumlah fonem)
  for (int t=0; t < batch_frame; t++) {
    int sym = m.outputs[unpack_ptr(h_path[t])];
    if(sym != EPS_SYM){
      output_symbols.push_back(t);
    }
  }

  return unpack_prob(h_path.back());
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

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


    gpu_fst m = read_fst_noNumberizer(fst_rspecifier) // harus berupa file text
    
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
      GPUAmDiagGmm gpu_am_gmm;
      for(size_t i = 0; i < am_gmm.NumPdfs(); ++i){
        GPUDiagGmm *gpu_gmm = new GPUDiagGmm(am_gmm.GetPdf(i));
        gpu_am_gmm.AddPdf(*gpu_gmm);
      }
      
      // Copy TransitionModel ke GPUTransitionModel
      GPUTransitionModel gpu_trans_model = GPUTransitionModel(trans_model);
      
      // Create GPUOnlineDecodableDiagGmmScaled
      GPUOnlineDecodableDiagGmmScaled gpu_decodable(gpu_am_gmm, gpu_trans_model, acoustic_scale);
      
      // TODO : Buat Loop Decodenya
      
      auto read_start = std::chrono::steady_clock::now();
      frame_ = 0;
      while(1){
        std::vector<sym_t> output_symbols;

        prob_t final_prob = viterbi(m, &decodable, &gpu_decodable, output_symbols);
        std::cout << onr.join(output_symbols) << " "; 
        // IF END BREAK;
      }
      std::cout << std::endl;
      auto read_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = read_end - read_start;
      std::cout << "Time to decode sentence: " << diff.count()  << std::endl;

      delete feat_transform;
    }
    delete word_syms;
    delete decode_fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
