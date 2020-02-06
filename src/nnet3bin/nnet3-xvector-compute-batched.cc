// nnet3bin/nnet3-xvector-compute.cc

// Copyright 2019   Daniel Povey
//           2017   Johns Hopkins University (author: Daniel Povey)
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


struct BatchedXvectorComputerOptions {
  int32 chunk_size { 150 };
  int32 batch_size { 32 };
  bool pad_input { true };
  NnetComputeOptions compute_config;
  NnetOptimizeOptions optimize_config;
  CachingOptimizingCompilerOptions compiler_config;


  void Register(OptionsItf *po) {
    po->Register("chunk-size", &chunk_size,
                 "Size of chunk, in input frames.  Includes the nnet "
                 "context, so the number of chunks will be more than "
                 "total-input-frames / chunk-size.");
    po->Register("batch-size", &batch_size,
                 "Size of the batches of chunks that we compute at once. ");
    po->Register("pad-input", &pad_input,
                 "If true, for utterances shorter than `chunk-size` frames "
                 "we will pad with repeats of the last frame.");
    compute_config.Register(po);
    optimize_config.Register(po);
    compiler_config.Register(po);
  }
};


/**
   This function divides the number 'a' into 'b' pieces, such that
   the sum of the pieces equals 'a' and no two pieces differ by more
   than 1.
     @param [in] a     A number, may be positive or negative
     @param [in] b     The number of pieces, b >= 1.
     @param [out] pieces   The pieces will be written to here.
                       At exit, their sum will equal a, and none
                       of them will differ from any other by more
                       than 1.  Otherwise they are arbitrarily
                       chosen.
 */
void DivideIntoPieces(int32 a, int32 b, std::vector<int32> *pieces) {
  KALDI_ASSERT(b > 0);
  pieces->clear();
  pieces->reserve(b);
  int32 a_sign = 1;
  // Make sure a is positive before division, because the behavior of division
  // with negative operands is not fully defined in C.
  if (a < 0) {
    a_sign = -1;
    a *= -1;
  }
  int32 piece_size1 = a / b,
      piece_size2 = piece_size1 + 1,
      remainder = a % b;
  int32 num_pieces_of_size1 = b - remainder,
      num_pieces_of_size2 = remainder;
  KALDI_ASSERT(a == num_pieces_of_size1 * piece_size1 +
               num_pieces_of_size2 * piece_size2);

  for (int32 i = 0; i < num_pieces_of_size1; i++)
    pieces->push_back(piece_size1 * a_sign);
  for (int32 i = 0; i < num_pieces_of_size2; i++)
    pieces->push_back(piece_size2 * a_sign);
}



class BatchedXvectorComputer {
 public:
  /**
       @param [in]  opts  Options class; warning, it keeps a reference to it.
       @param [in]  nnet  The neural net we'll be computing with; assumed to have
                          already been prepared for test.
       @param [in] total_context   The sum of the left and right context of the
                          network, computed after calling
                          SetRequireDirectInput(true, &nnet); so the l/r context
                          isn't zero.
   */

  BatchedXvectorComputer(const BatchedXvectorComputerOptions &opts,
                         const Nnet &nnet,
                         int32 total_context);

  /**
     Accepts an utterance to process into an xvector, and, if one or more
     batches become full, processes the batch.
   */
  void AcceptUtterance(const std::string &utt,
                      const Matrix<BaseFloat> &input);


  /**  Returns true if at least one xvector is pending output (i.e. that
       the user may call OutputXvector()).
   */
  bool XvectorReady() const;

  /**
     This function, which must only be called if XvectorReady() has
     just returned true,  outputs an xvector for an utterance.
       @param [out] utt  The utterance-id is written to here.
                        Note: these will be output in the same order
                        as the user called AcceptUtterance(), except
                        that if opts_.pad_input is false and
                        and utterance is shorter than the chunk
                        size, some utterances may be skipped.
       @param [out] xvector  The xvector will be written to here.
   */
  void OutputXvector(std::string *utt,
                     Vector<BaseFloat> *xvector);


  /**
     Calling this will force any partial minibatch to be computed,
     so that any utterances that have previously been passed to
     AcceptUtterance() will, when this function returns, have
     their xvectors ready to be retrieved by OutputXvector().
   */
  void Flush();


 private:

  struct XvectorTask {
    std::string utt_id;
    int32 num_chunks;
    int32 num_chunks_finished;
    Vector<BaseFloat> xvector;
    XvectorTask *tail;
  };


  /**
     This decides how to split the utterance into chunks.  It does so in a way
     that minimizes the variance of the x-vector under some simplifying
     assumptions.  It's about minimizing the variance of the x-vector.  We treat
     the x-vector as computed as a sum over frames (although some frames may be
     repeated or omitted due to gaps between chunks or overlaps between chunks);
     and we try to minimize the variance of the x-vector estimate; this is minimized
     when all the frames have the same weight, which is only possible if it can be
     exactly divided into chunks; anyway, this function computes the best division
     into chunks.

     It's a question of whether to allow overlaps or gaps.
     Suppose we are averaging independent quantities with variance 1.  The
     variance of a simple sum of M of those quantities is 1/M.
     Suppose we have M of those quantities, plus N which are repeated twice
     in the sum.  The variance of the estimate formed that way is:

      (M + 4N) / (M + 2N)^2

     If we can't divide it exactly into chunks we'll compare the variances from
     the cases where there is a gap vs. an overlap, and choose the one with
     the smallest variance.  (Note: due to context effects we actually lose
     total_context_ frames from the input signal, and the chunks would have
     to overlap by total_context_ even if the part at the statistics-computation
     layer were ideally cut up.

        @param [in] num_frames  The number of frames in the utterance
        @param [out] start_frames  This function will output to here a vector
                    containing all the start-frames of chunks in this utterance.
                    All chunks will have duration opts_.chunk_size; if a chunk
                    goes past the end of the input we'll repeat the last frame.
                    (This will only happen if opts_.pad_input is false and
                    num_frames is less than opts_.chunk_length.)
   */
  void SplitUtteranceIntoChunks(int32 num_frames,
                                std::vector<int32> *start_frames);

  /** This adds a newly created XvectorTask at the tail of the singly linked
      list whose (head,tail) are results_head_, results_tail_.
   */
  XvectorTask* CreateTask(const std::string &utt, int32 num_chunks);


  /**
     Does the nnet computation for one batch and distributes the
     computed x-vectors (of chunks) appropriately to their XvectorTask
     objects.
   */
  void ComputeOneBatch();

  /**
     Adds a new chunk to a batch we are preparing.  This will go
     at position `position_in_batch_` which will be incremented.
       @param [in] task  The task this is part of (records the
                utterance); tasks_this_batch_[position_in_batch_] will
                be set to this.
       @param [in] input  The input matrix of features of
                which this chunk is a part
       @param [in] chunk_start  The frame at which this
                chunk starts.  Must be >= 0; and if
                opts_.pad_input is false, chunk_start + opts_.chunk_size
                must be <= input.NumRows().
   */
  void AddChunkToBatch(XvectorTask *task,
                       const Matrix<BaseFloat> &input,
                       int32 chunk_start);

  const BatchedXvectorComputerOptions &opts_;
  int32 total_context_;
  const Nnet &nnet_;

  int32 feature_dim_;
  int32 xvector_dim_;

  /**
     Staging area for the input features prior to copying them to GPU.
     Dimension is opts_.chunk_size * opts_.batch_size by feature_dim_.  The
     sequences are interleaved (will be faster since this corresponds to how
     nnet3 keeps things in memory), i.e. row 0 of input_feats_ is time t=0
     for chunk n=0; and row 1 of input_feats_ is time t=0 for chunk n=1.
  */
  Matrix<BaseFloat> input_feats_;


  /** The compiled computation (will be the same for every batch).  */
  std::shared_ptr<const NnetComputation> computation_;


  /**  position_in_batch_ is the number of chunks that we have filled in in
       the input_feats_ matrix and tasks_this_batch_.  When it reaches
       opts_.batch_size we will do the actual computation.
  */
  int32 position_in_batch_;

  /**
     tasks_this_batch_ is of dimension opts_.batch_size.  It is a vector of pointers to
     elements of the singly linked list whose head is at results_head_, or
     NULL for elements with indexes >= position_in_batch_.
   */
  std::vector<XvectorTask*> tasks_this_batch_;

  // results_head_ is the first element in the singly linked list of
  // already-computed xvectors, or NULL if that list is empty.  Note:
  // utterances that are ready will appear here first; new utterances
  // get added to the tail.
  XvectorTask *results_head_;
  // results_tail_ is the last element in the singly linked list of
  // already-computed xvectors, or NULL if the list is empty.
  XvectorTask *results_tail_;
};

BatchedXvectorComputer::XvectorTask*
BatchedXvectorComputer::CreateTask(
    const std::string &utt, int32 num_chunks) {
  XvectorTask *task = new XvectorTask;
  task->utt_id = utt;
  task->num_chunks = num_chunks;
  task->num_chunks_finished = 0;
  task->xvector.Resize(xvector_dim_);
  task->tail = NULL;
  if (results_tail_) {
    results_tail_->tail = task;
    results_tail_ = task;
  } else {  // List was previously empty.
    results_head_ = task;
    results_tail_ = task;
  }
  return task;
}

BatchedXvectorComputer::BatchedXvectorComputer(
    const BatchedXvectorComputerOptions &opts,
    const Nnet &nnet,
    int32 total_context):
    opts_(opts),
    total_context_(total_context),
    nnet_(nnet),
    position_in_batch_(0),
    results_head_(NULL),
    results_tail_(NULL) {

  tasks_this_batch_.resize(opts_.batch_size);

  feature_dim_ = nnet.InputDim("input");
  xvector_dim_ = nnet.OutputDim("output");
  // Zero input_feats_ in case there is only one batch, to avoid
  // NaN's being generated due to undefined data.
  input_feats_.Resize(opts_.chunk_size * opts_.batch_size,
                      feature_dim_);

  CachingOptimizingCompiler compiler(nnet, opts.optimize_config,
                                     opts.compiler_config);

  {  // This block creates computation_.
    ComputationRequest request;
    request.need_model_derivative = false;
    request.store_component_stats = false;
    request.inputs.resize(1);
    IoSpecification &input(request.inputs[0]);
    input.name = "input";
    input.has_deriv = false;
    input.indexes.resize(opts_.batch_size * opts_.chunk_size);
    // Note: the sequences are interleaved in the input; this will save an extra
    // copy since it corresponds to how nnet3 stores things by default.  (Makes
    // TDNNs easier to implement.)
    for (int32 n = 0; n < opts_.batch_size; n++) {
      for (int32 t = 0; t < opts_.chunk_size; t++) {
        Index index;
        index.n = n;
        index.t = t;
        // index.x is 0 by default.
        input.indexes[n + opts_.batch_size * t] = index;
      }
    }
    IoSpecification output;
    output.name = "output";
    output.has_deriv = false;
    output.indexes.resize(opts_.batch_size);
    for (int32 n = 0; n < opts_.batch_size; n++){
        Index index;
        index.n = n;
        index.t = 0;
        output.indexes[n] = index;
    }
    request.outputs.push_back(output);
    computation_ = compiler.Compile(request);
  }
}

void BatchedXvectorComputer::AddChunkToBatch(
    XvectorTask *task,
    const Matrix<BaseFloat> &input,
    int32 chunk_start) {
  int32 n = position_in_batch_++;
  KALDI_ASSERT(n >= 0 && n < opts_.batch_size);
  tasks_this_batch_[n] = task;
  int32 T = opts_.chunk_size,
      num_input_frames = input.NumRows();
  KALDI_ASSERT(input_feats_.NumRows() == T * opts_.batch_size);
  if (input.NumCols() != feature_dim_) {
    KALDI_ERR << "Feature dimension mismatch: neural net expected "
              << feature_dim_ << ", got " << input.NumCols();
  }
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> dest(input_feats_, t * opts_.batch_size + n);
    int32 src_t = t + chunk_start;
    if (src_t >= num_input_frames) {
      KALDI_ASSERT(opts_.pad_input);
      src_t = num_input_frames - 1;  // Pad with repeats of the last frame.
    }
    SubVector<BaseFloat> src(input, src_t);
    dest.CopyFromVec(src);
  }
}

bool BatchedXvectorComputer::XvectorReady() const {
  if (results_head_ == NULL)
    return false;
  KALDI_ASSERT(results_head_->num_chunks_finished <= results_head_->num_chunks);
  return results_head_->num_chunks_finished == results_head_->num_chunks;
}

void BatchedXvectorComputer::OutputXvector(std::string *utt,
                                           Vector<BaseFloat> *xvector) {
  KALDI_ASSERT(XvectorReady());
  *utt = results_head_->utt_id;
  xvector->Swap(&(results_head_->xvector));
  XvectorTask *new_tail = results_head_->tail;
  delete results_head_;
  results_head_ = new_tail;
  if (new_tail == NULL)
    results_tail_ = NULL;
}

void BatchedXvectorComputer::Flush() {
  if (position_in_batch_ == 0)
    return;
  ComputeOneBatch();
}


void BatchedXvectorComputer::ComputeOneBatch() {

  CuMatrix<BaseFloat> cu_input_feats(input_feats_);
  Nnet *nnet_to_update = NULL;  // we're not doing any update.
  NnetComputer computer(opts_.compute_config, *computation_,
                        nnet_, nnet_to_update);
  computer.AcceptInput("input", &cu_input_feats);
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  KALDI_ASSERT(cu_output.NumRows() == opts_.batch_size);
  Matrix<BaseFloat> output(cu_output);
  for (int32 n = 0; n < opts_.batch_size; n++) {
    XvectorTask *task = tasks_this_batch_[n];
    if (task == NULL)
      continue;  // Would only happen for the last batch.
    task->num_chunks_finished++;
    task->xvector.AddVec(1.0 / task->num_chunks, output.Row(n));
  }
  position_in_batch_ = 0;
  std::fill(tasks_this_batch_.begin(), tasks_this_batch_.end(),
            (XvectorTask*)NULL);
}

void BatchedXvectorComputer::AcceptUtterance(
    const std::string &utt,
    const Matrix<BaseFloat> &input) {
  std::vector<int32> chunk_starts;
  int32 num_frames = input.NumRows();
  SplitUtteranceIntoChunks(num_frames, &chunk_starts);
  int32 num_chunks = chunk_starts.size();
  XvectorTask *task = CreateTask(utt, num_chunks);

  for (int32 i = 0; i < num_chunks; i++) {
    AddChunkToBatch(task, input, chunk_starts[i]);
    if (position_in_batch_ == opts_.batch_size) {
      ComputeOneBatch();
    }
  }
}

void BatchedXvectorComputer::SplitUtteranceIntoChunks(
    int32 num_frames, std::vector<int32> *start_frames) {
  start_frames->clear();
  if (num_frames <= opts_.chunk_size) {
    if (num_frames == opts_.chunk_size || opts_.pad_input)
      start_frames->push_back(0);
    // if we leave start_frames empty, then we just won't compute anything for
    // this file.
  } else {
    // these modified quantities are to account for the context effects...  when
    // the chunks overlap by exactly total_context_, the frames that get
    // averaged by the respective chunks in their averaging layers would touch
    // but not overlap.  So the optimal separation between chunks would equal
    // opts_.chunk_size - total_context_.
    int32 modified_num_frames = num_frames - total_context_,
        modified_chunk_size = opts_.chunk_size - total_context_;
    KALDI_ASSERT(modified_num_frames > modified_chunk_size);
    int32 num_chunks1 = modified_num_frames / modified_chunk_size,
        num_chunks2 = num_chunks1 + 1;
    int32 num_frames1 = num_chunks1 * modified_chunk_size,
        num_frames2 = num_chunks2 * modified_chunk_size;
    KALDI_ASSERT(num_frames2 > modified_chunk_size);
    // The M and N below correspond to the M and N in the comment:
    // M is the number of frames repeated once in the averaging, N
    // the number of frames repeated twice.  (Basically a solution
    // of the equations: (M + 2N == num_frames2, M+N == modified_num_frames).
    // Note: by a "frame" above, I mean a specific "t" value in
    // the utterance.
    int32 N = num_frames2 - modified_num_frames,
        M = modified_num_frames - N;
    KALDI_ASSERT(M + 2*N == num_frames2 && M + N == modified_num_frames);

    // The variances below are proportional to the variance of our
    // estimate of the xvector under certain simplifying assumptions..
    // they help us choose whether to have gaps between the chunks
    // or overlaps between them.
    BaseFloat variance1 = 1.0 / num_frames1,  // the 1/M mentioned above.
        variance2 = (M + 4.0*N) / ((M + 2.0*N)*(M + 2.0*N));
    if (variance1 <= variance2) {
      // We'll choose the smaller number of chunks.  There may be gaps.
      // Counting the positions at the ends, there are num_chunks+1 positions
      // where there might be gaps.
      // Note: "total_gap" is >= 0, it's the positive of the sum of the
      // sizes of those gaps.
      int32 num_chunks = num_chunks1,
          num_gaps = num_chunks + 1,
          total_gap = modified_num_frames - num_chunks * modified_chunk_size;
      KALDI_ASSERT(0 <= total_gap && total_gap < modified_chunk_size);
      std::vector<int32> gap_sizes;  // elements will be >= 0.
      DivideIntoPieces(total_gap, num_gaps, &gap_sizes);
      int32 pos = gap_sizes[0];
      for (int32 i = 0; i < num_chunks; i++) {
        start_frames->push_back(pos);
        pos += modified_chunk_size + gap_sizes[i + 1];
      }
      KALDI_ASSERT(pos == modified_num_frames);
    } else {
      int32 num_chunks = num_chunks2,
          num_overlaps = num_chunks - 1,
          total_overlap = modified_num_frames - num_chunks * modified_chunk_size;
      KALDI_ASSERT( -modified_chunk_size < total_overlap && total_overlap <= 0 );
      std::vector<int32> overlap_sizes;  // elements will be <= 0.
      DivideIntoPieces(total_overlap, num_overlaps, &overlap_sizes);
      int32 pos = 0;
      for (int32 i = 0; i < num_chunks; i++) {
        start_frames->push_back(pos);
        pos += modified_chunk_size;
        if (i < num_overlaps)
          pos += overlap_sizes[i];
      }
      KALDI_ASSERT(pos == modified_num_frames);
    }
  }
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

    BatchedXvectorComputerOptions opts;

    std::string use_gpu = "no";

    opts.Register(&po);

    po.Register("use-gpu", &use_gpu,
      "yes|no|optional|wait, only has effect if compiled with CUDA");

#if HAVE_CUDA==1
    CuDevice::RegisterDeviceOptions(&po);
#endif
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

    int32 total_context;
    {
      int32 left_context, right_context;
      // Compute left_context, right_context as the 'real' left/right context
      // of the network; they'll tell us how many frames on the chunk boundaries
      // won't really participate in the statistics averaging.
      // SetRequireDirectInput()  modifies how the StatisticsPoolingComponent
      // treats its dependences, so we'll get the 'real' left/right context.
      SetRequireDirectInput(true, &nnet);
      ComputeSimpleNnetContext(nnet, &left_context, &right_context);
      KALDI_LOG << "Left/right context is " << left_context << ", "
                << right_context;
      SetRequireDirectInput(false, &nnet);
      total_context = left_context + right_context;
    }

    BatchedXvectorComputer computer(opts, nnet, total_context);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_utts_read = 0, num_xvectors_written = 0;
    int64 frame_count = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        continue;
      }

      frame_count += features.NumRows();

      computer.AcceptUtterance(utt, features);
      num_utts_read++;

      while (computer.XvectorReady()) {
        std::string utt;
        Vector<BaseFloat> xvector;
        computer.OutputXvector(&utt, &xvector);
        vector_writer.Write(utt, xvector);
        num_xvectors_written++;
      }
    }

    computer.Flush();
    while (computer.XvectorReady()) {
      std::string utt;
      Vector<BaseFloat> xvector;
      computer.OutputXvector(&utt, &xvector);
      vector_writer.Write(utt, xvector);
      num_xvectors_written++;
    }


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Read " << num_utts_read << " utterances, wrote "
              << num_xvectors_written << " xvectors.";

    // Note: the following rule does something reasonable even if there are 0, 1
    // or 2 utterances read.
    if (num_xvectors_written > num_utts_read / 2)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
