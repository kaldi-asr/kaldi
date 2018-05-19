// This is the main program of Radit's Final Project
// The file mostly contains the same program with onlinebin/online-wav-gmm-decode-faster.cpp

// Uses Parallel Viterbi Beam Search Algorithm from Arturo Argueta and David Chiang's paper.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "base/kaldi-math.h"
#include "gmm/diag-gmm.h"

#include "gpufst/fst.h"
#include "gpufst/gpu-fst.h"
#include "gpufst/prob-ptr.h"
#include "gpufst/numberizer.h"

#include "gpucommons/gpu-vector.h"
#include "gpucommons/gpu-matrix.h"
#include "hmm/transition-model.h"
#include "matrix/matrix-common.h"


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

#include <math_functions.h>
#include <math_constants.h>

int ceildiv(int x, int y) { return (x-1)/y+1; }
#define BLOCK_SIZE 512
#define BEAM_SIZE 10
#define BATCH_SIZE 216
#define LIKELIHOOD_BLOCK_SIZE 128

const int NUM_EPS_LAYER = 1;
const int NUM_LAYER = NUM_EPS_LAYER + 1;
const int NUM_BIT_LAYER = __builtin_popcount(NUM_EPS_LAYER);
const int NUM_BIT_SHL_LAYER = 31 - NUM_BIT_LAYER;

const int NUM_FEATURE_ALLOCATE = 100;
const int NUM_GCONSTS_ALLOCATE = 50;
const int NUM_BUFFER_ALLOCATE = NUM_FEATURE_ALLOCATE + NUM_GCONSTS_ALLOCATE;

const int OFFSET_AND_BIT = (1 << NUM_BIT_SHL_LAYER) - 1;

#define EPS_SYM 0

namespace kaldi{

__device__
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown status";
}

__device__ cublasStatus_t cublas_gemv_gpu(
    cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, float* alpha, const float* A, int lda, const float* x,
    int incx, float* beta, float* y, int incy) {
  return cublasSgemv_v2(handle,trans,m,n, alpha,A,lda,x,incx, beta,y,incy);
}

__device__ cublasStatus_t cublas_gemv_gpu(
    cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, double* alpha, const double* A, int lda, const double* x,
    int incx, double* beta, double* y, int incy) {
  return cublasDgemv_v2(handle,trans,m,n, alpha,A,lda,x,incx, beta,y,incy);
}

__device__ cublasStatus_t AddMatVecGPU(
                  cublasHandle_t handle,
                  BaseFloat* data, 
                  BaseFloat* alpha,
                  const GPUMatrix<BaseFloat> &M,
                  MatrixTransposeType trans,
                  BaseFloat* v,  
                  BaseFloat* beta){
  return cublas_gemv_gpu(handle,
      (trans==kTrans? CUBLAS_OP_N:CUBLAS_OP_T),
      M.NumCols(), M.NumRows(), alpha, M.Data(),
      M.Stride(), v, 1, beta, data, 1);
}

struct GPUDiagGmm{
  GPUVector<BaseFloat> gconsts_;
  GPUVector<BaseFloat> weights_;
  GPUMatrix<BaseFloat> inv_vars_;
  GPUMatrix<BaseFloat> means_invvars_;

  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false

  GPUDiagGmm() {}
  GPUDiagGmm(DiagGmm &d) : 
    valid_gconsts_(d.valid_gconsts()),
    gconsts_(d.gconsts()),
    weights_(d.weights()),
    inv_vars_(d.inv_vars()),
    means_invvars_(d.means_invvars()) {}

  __device__
  int32 Dim() const { return means_invvars_.NumCols();}

  __device__
  BaseFloat LogLikelihood(BaseFloat *data, int32 num_data, BaseFloat* loglikes, BaseFloat* data_sq) {
    const double kGPUMinLogDiffDouble = log(DBL_EPSILON);
    const float kGPUMinLogDiffFloat = log(FLT_EPSILON);

    int32 num_loglikes = gconsts_.Dim();
    for(int32 i = 0;i < num_loglikes; ++i) loglikes[i] = gconsts_.data[i];

    for(int32 i = 0;i < num_data; ++i) data_sq[i] = data[i] * data[i];

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
      printf("CUBLAS CALL ERROR 0 : %s\n", cublasGetErrorString(stat));
      for(int i = 0;i < gconsts_.Dim(); ++i){
        for(int j = 0;j < num_data; ++j){
          loglikes[i] += means_invvars_.data[means_invvars_.Index(i, j)] * data[j];
          loglikes[i] -= 0.5 * inv_vars_.data[inv_vars_.Index(i, j)] * data_sq[j];
        }
      }
    }
    else {
      BaseFloat* beta = new BaseFloat(1.0);
      BaseFloat* alpha_1 = new BaseFloat(1.0);
      BaseFloat* alpha_2 = new BaseFloat(-0.5);

      stat = AddMatVecGPU(handle, loglikes, alpha_1, means_invvars_, kNoTrans, data, beta);
      if(stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS CALL ERROR 1 : %s\n", cublasGetErrorString(stat));
      }
      stat = AddMatVecGPU(handle, loglikes, alpha_2, inv_vars_, kNoTrans, data_sq, beta);
      if(stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS CALL ERROR 2 : %s\n", cublasGetErrorString(stat));
      }
      delete alpha_1;
      delete alpha_2;
      delete beta;
      cublasDestroy(handle); 
    }

    BaseFloat max_elem = -CUDART_MAX_NORMAL_F;
    for(int32 i = 0;i < num_loglikes; ++i) {
      if(max_elem < loglikes[i]) max_elem = loglikes[i];
    }

    BaseFloat cutoff;
    if (sizeof(BaseFloat) == 4) cutoff = max_elem + kGPUMinLogDiffFloat;
    else cutoff = max_elem + kGPUMinLogDiffDouble;
    double sum_relto_max_elem = 0.0;

    for (int32 i = 0; i < num_loglikes; i++) {
      BaseFloat f = loglikes[i];
      if (f >= cutoff)
        sum_relto_max_elem += exp(f - max_elem);
    }

    BaseFloat log_sum = max_elem + log(sum_relto_max_elem);

    return log_sum;
  }

};

struct GPUAmDiagGmm{

  thrust::device_vector<GPUDiagGmm*> densities_;
  GPUDiagGmm** densities;

  GPUAmDiagGmm() { densities = densities_.data().get(); }

  __device__ BaseFloat LogLikelihood(const int32 pdf_index, BaseFloat* data, int32 num_data, BaseFloat* buffers) {
    return densities[pdf_index]->LogLikelihood(data, num_data, buffers, buffers + NUM_FEATURE_ALLOCATE);
  }
};

struct GPUTransitionModel{

  thrust::device_vector<int32> id2pdf_id_;
  int32* id2pdf_id;
  int32 num_pdfs_;

  GPUTransitionModel() {}
  GPUTransitionModel(TransitionModel& t) :   
    id2pdf_id_(t.id2pdf_id()),
    num_pdfs_(t.NumPdfs())
  {
    id2pdf_id = id2pdf_id_.data().get();
  }

  __host__ __device__ int32 NumPdfs() const { return num_pdfs_;}

  __device__ int32 TransitionIdToPdf(int32 trans_id) const { 
    return id2pdf_id[trans_id];
  }

};

struct GPUOnlineDecodableDiagGmmScaled {

  // nggak perlu masukin OnlineFeatureMatrixnya jadinya (parsial aja)

  GPUAmDiagGmm* ac_model_;
  GPUTransitionModel* transition_model_;
  BaseFloat ac_scale_;
  int32 feat_dim_;

  int32 cur_frame_;

  GPUOnlineDecodableDiagGmmScaled() {}
  GPUOnlineDecodableDiagGmmScaled(
    GPUAmDiagGmm* gpu_ac_model_,
    GPUTransitionModel* gpu_transition_model_,
    BaseFloat ac_scale_
  ) : 
  ac_model_(gpu_ac_model_),
  transition_model_(gpu_transition_model_),
  ac_scale_(ac_scale_) {}
};

}

using namespace kaldi;
using namespace gpufst;

int frame_;
int utt_frames_;

__global__ void compute_initial(int *from_states, int *to_states, float *probs, int* inputs,
        int start_offset, int end_offset,
        state_t initial_state,
        prob_ptr_t *viterbi, int num_states)
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
    atomicMax(&viterbi[to_shared_states[threadIdx.x]], pack(-shared_probs[threadIdx.x], offset));
  }

  viterbi[initial_state] = pack(0.0f, 0);

  __syncthreads();
  if (offset < end_offset){
    for(int i = 1;i <= NUM_EPS_LAYER; ++i){
      viterbi[to_shared_states[threadIdx.x] + num_states * i] = viterbi[to_shared_states[threadIdx.x]];
    }    
  }
  if (id == 0){
    for(int i = 1;i <= NUM_EPS_LAYER; ++i){
      viterbi[initial_state + num_states * i] = viterbi[initial_state];
    }
  }
}


__global__ void compute_loglikelihoods(int frame, 
  GPUOnlineDecodableDiagGmmScaled* decodable, 
  float* loglikelihoods, int max_pdf_id,
  float* cur_feats, int cur_feats_dim, float* buffers){

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < max_pdf_id){
    loglikelihoods[id] = decodable->ac_model_->LogLikelihood(id, cur_feats, cur_feats_dim, buffers + id * NUM_BUFFER_ALLOCATE) * decodable->ac_scale_;
  }

} 

__global__ void compute_emitting(int *from_states, int *to_states, float *probs, int* inputs,
  int start_offset, int end_offset, int frame,
  prob_ptr_t *viterbi_prev, prob_ptr_t *viterbi,
  GPUOnlineDecodableDiagGmmScaled* decodable, float* loglikelihoods) {

  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int offset = start_offset+id;

  __shared__ int from_shared_states[BLOCK_SIZE];
  __shared__ int to_shared_states[BLOCK_SIZE];
  __shared__ prob_ptr_t viterbi_from_shared_states[BLOCK_SIZE];
  __shared__ float shared_probs[BLOCK_SIZE];
  __shared__ sym_t shared_inputs[BLOCK_SIZE];
 
  from_shared_states[threadIdx.x] = from_states[offset];
  to_shared_states[threadIdx.x] = to_states[offset];
  shared_probs[threadIdx.x] = probs[offset];
  shared_inputs[threadIdx.x] = inputs[offset];
  
  viterbi_from_shared_states[threadIdx.x] = viterbi_prev[from_shared_states[threadIdx.x]];

  if (offset < end_offset && shared_inputs[threadIdx.x] != EPS_SYM) {
    int idxlayer = ((unpack_ptr(viterbi_from_shared_states[threadIdx.x]) >> NUM_BIT_SHL_LAYER) + 1) & NUM_EPS_LAYER; // dapat index layernya keberapa
    float ac_cost = - loglikelihoods[decodable->transition_model_->TransitionIdToPdf(shared_inputs[threadIdx.x])];
    prob_ptr_t pp = pack(unpack_prob(viterbi_from_shared_states[threadIdx.x]) - shared_probs[threadIdx.x] - ac_cost, offset | (idxlayer << NUM_BIT_SHL_LAYER));
    atomicMax(&viterbi[to_shared_states[threadIdx.x]], pp);
  }
}

__global__ void compute_nonemitting(int *from_states, int *to_states, float *probs, int* inputs,
  int start_offset, int end_offset,                
  prob_ptr_t *viterbi_prev, prob_ptr_t *viterbi, int layer_idx){

   int id = blockIdx.x*blockDim.x+threadIdx.x;
  int offset = start_offset+id;

  __shared__ int from_shared_states[BLOCK_SIZE];
  __shared__ int to_shared_states[BLOCK_SIZE];
  __shared__ prob_ptr_t viterbi_from_shared_states[BLOCK_SIZE];
  __shared__ float shared_probs[BLOCK_SIZE];
  __shared__ sym_t shared_inputs[BLOCK_SIZE];

  from_shared_states[threadIdx.x] = from_states[offset];
  to_shared_states[threadIdx.x] = to_states[offset];
  shared_probs[threadIdx.x] = probs[offset];
  shared_inputs[threadIdx.x] = inputs[offset];
  viterbi_from_shared_states[threadIdx.x] = viterbi_prev[from_shared_states[threadIdx.x]];

  if (offset < end_offset){
    atomicMax(&viterbi[to_shared_states[threadIdx.x]], viterbi_prev[to_shared_states[threadIdx.x]]);
    if(shared_inputs[threadIdx.x] == EPS_SYM) {
      prob_ptr_t pp = pack(unpack_prob(viterbi_from_shared_states[threadIdx.x]) - shared_probs[threadIdx.x], offset | ((layer_idx - 1) << NUM_BIT_SHL_LAYER));
      atomicMax(&viterbi[to_shared_states[threadIdx.x]], pp);
    }
  }
}

__global__ void compute_max(
           prob_ptr_t *viterbi_prev, 
           prob_ptr_t *max_prob,
           int num_states) {
  // To do: this should be a reduction instead
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < num_states) {
    prob_ptr_t pp = pack(unpack_prob(viterbi_prev[id]), id);
    atomicMax(max_prob, pp);
  }
}

__global__ void get_path(int *from_nodes,
            prob_ptr_t *viterbi,
            int batch_frame, int num_nodes,
            prob_ptr_t *path, int* num_path) {
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int num_path_d = 0;
  if (id == 0) {
    int state = unpack_ptr(path[(batch_frame + 1) * NUM_LAYER]);
    float nilai = unpack_prob(path[(batch_frame + 1) * NUM_LAYER]);
    printf("STATE MAX : %d, nilai : %.10f\n", state, nilai);
    for (int t = batch_frame; t > 0; num_path_d++, t--) {
      while(state >= (1 << NUM_BIT_SHL_LAYER)){
        prob_ptr_t pp = viterbi[(t * NUM_LAYER + (state >> NUM_BIT_SHL_LAYER)) * num_nodes + (state & OFFSET_AND_BIT)];
        path[num_path_d] = pp; num_path_d++;
        state = from_nodes[unpack_ptr(pp) & OFFSET_AND_BIT] + ((unpack_ptr(pp) >> NUM_BIT_SHL_LAYER) << NUM_BIT_SHL_LAYER);
      }
      prob_ptr_t pp = viterbi[t * NUM_LAYER * num_nodes+state];
      path[num_path_d] = pp;
      state = from_nodes[unpack_ptr(pp) & OFFSET_AND_BIT] + ((unpack_ptr(pp) >> NUM_BIT_SHL_LAYER) << NUM_BIT_SHL_LAYER);
    }
    *num_path = num_path_d;
  }
}

/* TODO
 * 1. (DONE) Ganti Input Symbols jadi decodablenya
 * 2. (DONE) Ganti Resizenya jadi Batch Size
 * 3. (DONE) Konsep Beam Search disini beda, disini prune banyak state (sama dengan --max-active-state), di kaldi max prob
 * 4. (DONE) Input Offsets kayaknya gaperlu, semuanya dipake sekarang soalnya
 * 5. (DONE) Kasih offset frame di fungsi utama
 * 6. (DONE) Compute Final ga harus dari final state. kalo misalnya dia ga nyampe final state maka cari yang terbaik.
 * 7. (DONE) Ganti yang dapet output symbol dari input symbol. SIZE ga harus sama.
 * 8. (KAYAKNYA GA DEH) Abis batch frame, itu ga harus dari m.initial, ini harus coba dicari lagi mulainya dari mana.
 */
int viterbi(gpu_fst &m, 
  OnlineDecodableDiagGmmScaled* decodable, 
  GPUOnlineDecodableDiagGmmScaled* gpu_decodable, 
  int NUM_PDFS, 
  std::vector<sym_t> &output_symbols) {
  int verbose=0;

  int batch_frame = 0;

  static thrust::device_vector<prob_ptr_t> viterbi;
  viterbi.resize((BATCH_SIZE + 1) * m.num_states * NUM_LAYER);
  prob_ptr_t init_value = pack(-FLT_MAX, 0);
  thrust::fill(viterbi.begin(), viterbi.end(), init_value);

  static thrust::device_vector<prob_ptr_t> path((BATCH_SIZE + 1) * NUM_LAYER + 1);
  thrust::fill(path.begin(), path.end(), init_value);

  static thrust::device_vector<float> loglikelihoods(NUM_PDFS);
  static thrust::device_vector<float> buffers(NUM_BUFFER_ALLOCATE * NUM_PDFS);

  int start_offset = 0; 
  int end_offset = m.input_offsets.back();
  int BLOCKS = ceildiv(end_offset-start_offset, BLOCK_SIZE);
  int LIKELIHOOD_BLOCKS = ceildiv(NUM_PDFS, LIKELIHOOD_BLOCK_SIZE);
  compute_initial <<<BLOCKS, BLOCK_SIZE>>> (
    m.from_states.data().get(),
    m.to_states.data().get(),
    m.probs.data().get(),
    m.inputs.data().get(),
    start_offset, end_offset,
    m.initial,
    viterbi.data().get(),
    m.num_states
  );

  if (verbose) {
    for (auto pp: viterbi)
      if(unpack_prob(pp) != -FLT_MAX) std::cout << unpack_prob(pp) << " " << unpack_ptr(pp) << std::endl;
    std::cout << std::endl;
  }

  for (int t = NUM_LAYER; !decodable->IsLastFrame(frame_ - 1) && batch_frame < BATCH_SIZE;
     ++frame_, ++utt_frames_, ++batch_frame, t += NUM_LAYER) {
   
    decodable->CacheFrameFromGPU(frame_);
    GPUVector<BaseFloat> gpu_cur_feats(decodable->cur_feats());
    // thrust::copy(gpu_cur_feats.data_.begin(), gpu_cur_feats.data_.end(), std::ostream_iterator<float>(std::cout, " "));

    compute_loglikelihoods<<<LIKELIHOOD_BLOCKS, LIKELIHOOD_BLOCK_SIZE>>> (
      frame_, 
      gpu_decodable, 
      loglikelihoods.data().get(), 
      NUM_PDFS,
      gpu_cur_feats.data,
      gpu_cur_feats.Dim(),
      buffers.data().get()
    );

    cudaDeviceSynchronize();

    
    // compute nonepsilon / emitting
    compute_emitting <<<BLOCKS, BLOCK_SIZE>>> (
      m.from_states.data().get(), 
      m.to_states.data().get(),
      m.probs.data().get(),
      m.inputs.data().get(),
      start_offset, end_offset, 
      frame_,
      viterbi.data().get() + (t-1)*m.num_states,
      viterbi.data().get() + t*m.num_states, 
      gpu_decodable, loglikelihoods.data().get());

    // compute epsilon / nonemitting, banyaknya loop tergantung banyaknya layer propagasi
    for(int i = 1; i <= NUM_EPS_LAYER; ++i){
      compute_nonemitting <<<BLOCKS, BLOCK_SIZE>>> (
        m.from_states.data().get(), 
        m.to_states.data().get(),
        m.probs.data().get(),
        m.inputs.data().get(),
        start_offset, end_offset, 
        viterbi.data().get() + (t + i - 1) * m.num_states,
        viterbi.data().get() + (t + i) * m.num_states,
        i
      );
    }

    cudaDeviceSynchronize();
    
    if (verbose) {
      for (auto pp: viterbi)
        if(unpack_prob(pp) != -FLT_MAX) std::cout << unpack_prob(pp) << " " << unpack_ptr(pp) << std::endl; 
      std::cout << std::endl;
    }
   
  }

  compute_max<<<ceildiv(end_offset - start_offset, 1024), 1024>>> (
    viterbi.data().get() + ((batch_frame + 1) * NUM_LAYER - 1) * m.num_states,
    (&path.back()).get(),
    m.num_states
  );

  int* num_path_d;
  cudaMalloc((void**) &num_path_d, sizeof(int));
  get_path <<<1,1>>> (
    m.from_states.data().get(),
    viterbi.data().get(), 
    batch_frame, m.num_states,
    path.data().get(),
    num_path_d
  );
  int num_path;
  
  cudaMemcpy(&num_path, num_path_d, sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(num_path_d);
  cudaError_t e = cudaGetLastError();                                 
  if (e != cudaSuccess) {                                              
    std::cerr << "CUDA failure: " << cudaGetErrorString(e) << std::endl;
    exit(1);
  }
  
  thrust::host_vector<prob_ptr_t> h_path(path);
  output_symbols.clear();

  for (int t= num_path - 1; t >= 0; t--) {

    int sym = m.outputs[unpack_ptr(h_path[t]) & OFFSET_AND_BIT];
    if(sym != EPS_SYM){
      output_symbols.push_back(sym);
    }
  }
  
  if(batch_frame != BATCH_SIZE){
    return 2;
  }
  else{
    return 1;
  }
}

__global__ void AddPdfToGPUAmGmm(GPUAmDiagGmm* gpu_am_gmm, int pdf_idx, GPUDiagGmm *gpu_gmm){
  gpu_am_gmm->densities[pdf_idx] = gpu_gmm;
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
    
    const numberizer onr = read_numberizer(word_syms_filename);
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

      std::vector<GPUDiagGmm*> gpu_gmm_hs;
      GPUDiagGmm* gpu_gmm_h;
      for(size_t i = 0; i < am_gmm.NumPdfs(); ++i){
        gpu_gmm_h = new GPUDiagGmm(am_gmm.GetPdf(i));
        gpu_gmm_hs.push_back(gpu_gmm_h);

        GPUDiagGmm *gpu_gmm_d;
        cudaMalloc((void**) &gpu_gmm_d, sizeof(GPUDiagGmm));
        cudaMemcpy(gpu_gmm_d, gpu_gmm_h, sizeof(GPUDiagGmm), cudaMemcpyHostToDevice);
        AddPdfToGPUAmGmm<<<1,1>>>(gpu_am_gmm_d, i, gpu_gmm_d);
      }

      // Copy TransitionModel ke GPUTransitionModel
      
      GPUTransitionModel *gpu_trans_model_h = new GPUTransitionModel(trans_model);
      GPUTransitionModel *gpu_trans_model_d;
      
      cudaMalloc((void**) &gpu_trans_model_d, sizeof(GPUTransitionModel));
      cudaMemcpy(gpu_trans_model_d, gpu_trans_model_h, sizeof(GPUTransitionModel), cudaMemcpyHostToDevice); 
      
      // Create GPUOnlineDecodableDiagGmmScaled
      GPUOnlineDecodableDiagGmmScaled* gpu_decodable_h = new GPUOnlineDecodableDiagGmmScaled(gpu_am_gmm_d, gpu_trans_model_d, acoustic_scale);
      GPUOnlineDecodableDiagGmmScaled* gpu_decodable_d;
      cudaMalloc((void**) &gpu_decodable_d, sizeof(GPUOnlineDecodableDiagGmmScaled));
      cudaMemcpy(gpu_decodable_d, gpu_decodable_h, sizeof(GPUOnlineDecodableDiagGmmScaled), cudaMemcpyHostToDevice);

      /* MAIN DECODE*/
      auto read_start = std::chrono::steady_clock::now();
      frame_ = 0;
      while(1){
        std::cerr << "FRAME : " << frame_ << std::endl;
        std::vector<sym_t> output_symbols;

        int state = viterbi(m, &decodable, gpu_decodable_d, gpu_trans_model_h->NumPdfs(), output_symbols);
        std::cerr << "KELUAR DONG" << std::endl;
        std::cerr << "OUTPUT SYMBOLS SIZE : " << output_symbols.size() << std::endl; 
        for(size_t j = 0;j < output_symbols.size(); ++j){
          std::cerr << output_symbols[j] << " ";
        }
        std::cerr << std::endl;
        std::cout << onr.join(output_symbols) << " "; 
        if(state == 2) break;
      }
      std::cout << std::endl;
      auto read_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = read_end - read_start;
      std::cout << "Time to decode sentence: " << diff.count()  << std::endl;
      /* END DECODE */

      // TODO : BARU MASUKIN VITERBINYA 
      cudaFree(gpu_decodable_d);
      cudaFree(gpu_trans_model_d);
      cudaFree(gpu_am_gmm_d);
      
      delete gpu_decodable_h;
      delete gpu_trans_model_h;
      for(size_t i = 0;i < gpu_gmm_hs.size(); ++i) delete gpu_gmm_hs[i];
      gpu_gmm_hs.clear();
      delete feat_transform;

    }

    std::cerr << "SAMPE DI AKHIR BANGET" << std::endl;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
