// online2/online-nnet2-decoding-threaded.h

// Copyright 2014-2015  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_NNET2_DECODING_THREADED_H_
#define KALDI_ONLINE2_ONLINE_NNET2_DECODING_THREADED_H_

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "decoder/decodable-matrix.h"
#include "nnet2/am-nnet.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "util/kaldi-semaphore.h"

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{


/**
   class ThreadSynchronizer acts to guard an arbitrary type of buffer between a
   producing and a consuming thread (note: it's all symmetric between the two
   thread types).  It has a similar interface to a mutex, except that instead of
   just Lock and Unlock, it has Lock, UnlockSuccess and UnlockFailure, and each
   function takes an argument kProducer or kConsumer to identify whether the
   producing or consuming thread is waiting.

   The basic concept is that you lock the object; and if you discover the you're
   blocked because you're either trying to read an empty buffer or trying to
   write to a full buffer, you unlock with UnlockFailure; and this will cause
   your next call to Lock to block until the *other* thread has called Lock and
   then UnlockSuccess.  However, if at that point the other thread calls Lock
   and then UnlockFailure, it is an error because you can't have both producing
   and consuming threads claiming that the buffer is full/empty.  If you lock
   the object and were successful you call UnlockSuccess; and you call
   UnlockSuccess even if, for your own reasons, you ended up not changing the
   state of the buffer.
*/
class ThreadSynchronizer {
 public:
  ThreadSynchronizer();

  // Most calls to this class should provide the thread-type of the caller,
  // producing or consuming.  Actually the behavior of this class is symmetric
  // between the two types of thread.
  enum ThreadType { kProducer, kConsumer };

  // All functions returning bool will return true normally, and false if
  // SetAbort() was set; if they return false, you should probably call SetAbort()
  // on any other ThreadSynchronizer classes you are using and then return from
  // the thread.

  // call this to lock the object being guarded.
  bool Lock(ThreadType t);

  // Call this to unlock the object being guarded, if you don't want the next call to
  // Lock to stall.
  bool UnlockSuccess(ThreadType t);

  // Call this if you want the next call to Lock() to stall until the other
  // (producer/consumer) thread has locked and then unlocked the mutex.  Note
  // that, if the other thread then calls Lock and then UnlockFailure, this will
  // generate a printed warning (and if repeated too many times, an exception).
  bool UnlockFailure(ThreadType t);

  // Sets abort_ flag so future calls will return false, and future calls to
  // Lock() won't lock the mutex but will immediately return false.
  void SetAbort();

  ~ThreadSynchronizer();

 private:
  bool abort_;
  bool producer_waiting_;  // true if producer is/will be waiting on semaphore
  bool consumer_waiting_;  // true if consumer is/will be waiting on semaphore
  std::mutex mutex_;  // Locks the buffer object.
  ThreadType held_by_;  // Record of which thread is holding the mutex (if
                        // held); else undefined.  Used for validation of input.
  Semaphore producer_semaphore_;  // The producer thread waits on this semaphore
  Semaphore consumer_semaphore_;  // The consumer thread waits on this semaphore
  KALDI_DISALLOW_COPY_AND_ASSIGN(ThreadSynchronizer);
};




// This is the configuration class for SingleUtteranceNnet2DecoderThreaded.  The
// actual command line program requires other configs that it creates
// separately, and which are not included here: namely,
// OnlineNnet2FeaturePipelineConfig and OnlineEndpointConfig.
struct OnlineNnet2DecodingThreadedConfig {

  LatticeFasterDecoderConfig decoder_opts;

  BaseFloat acoustic_scale;

  int32 max_buffered_features;  // maximum frames of features we allow to be
                                // held in the feature buffer before we block
                                // the feature-processing thread.

  int32 feature_batch_size;  // maximum number of frames at a time that we decode
                             // before unlocking the mutex.  The only real cost
                             // here is a mutex lock/unlock, so it's OK to make
                             // this fairly small.
  int32 max_loglikes_copy;   // maximum unused frames of log-likelihoods we will
                             // copy from the decodable object back into another
                             // matrix to be supplied to the decodable object.
                             // make this too large-> will block the
                             // decoder-search thread while copying; too small
                             // -> the nnet-evaluation thread may get blocked
                             // for too long while waiting for the decodable
                             // thread to be ready.
  int32 nnet_batch_size;    // batch size (number of frames) we evaluate in the
                            // neural net, if this many is available.  To take
                            // best advantage of BLAS, you may want to set this
                            // fairly large, e.g. 32 or 64 frames.  It probably
                            // makes sense to tune this a bit.
  int32 decode_batch_size;  // maximum number of frames at a time that we decode
                            // before unlocking the mutex.  The only real cost
                            // here is a mutex lock/unlock, so it's OK to make
                            // this fairly small.

  OnlineNnet2DecodingThreadedConfig() {
    acoustic_scale = 0.1;
    max_buffered_features = 100;
    feature_batch_size = 2;
    nnet_batch_size = 32;
    max_loglikes_copy = 20;
    decode_batch_size = 2;
  }

  void Check();

  void Register(OptionsItf *opts) {
    decoder_opts.Register(opts);
    opts->Register("acoustic-scale", &acoustic_scale, "Scale used on acoustics "
                   "when decoding");
    opts->Register("max-buffered-features", &max_buffered_features, "Obscure "
                   "setting, affects multi-threaded decoding.");
    opts->Register("feature-batch-size", &max_buffered_features, "Obscure "
                   "setting, affects multi-threaded decoding.");
    opts->Register("nnet-batch-size", &nnet_batch_size, "Maximum batch size "
                   "(in frames) used when evaluating neural net likelihoods");
    opts->Register("max-loglikes-copy", &max_loglikes_copy,  "Obscure "
                   "setting, affects multi-threaded decoding.");
    opts->Register("decode-batch-sie", &decode_batch_size, "Obscure "
                   "setting, affects multi-threaded decoding.");
  }
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.  Each time this
   class is created, it creates three background threads, and the feature
   extraction, neural net evaluation, and search aspects of decoding all
   happen in different threads.
   Note: we assume that all calls to its public interface happen from a single
   thread.
*/
class SingleUtteranceNnet2DecoderThreaded {
 public:
  // Constructor.  Unlike SingleUtteranceNnet2Decoder, we create the
  // feature_pipeline object inside this class, since access to it needs to be
  // controlled by a mutex and this class knows how to handle that.  The
  // feature_info and adaptation_state arguments are used to initialize the
  // (locally owned) feature pipeline.
  SingleUtteranceNnet2DecoderThreaded(
      const OnlineNnet2DecodingThreadedConfig &config,
      const TransitionModel &tmodel,
      const nnet2::AmNnet &am_nnet,
      const fst::Fst<fst::StdArc> &fst,
      const OnlineNnet2FeaturePipelineInfo &feature_info,
      const OnlineIvectorExtractorAdaptationState &adaptation_state,
      const OnlineCmvnState &cmvn_state);



  /// You call this to provide this class with more waveform to decode.  This
  /// call is, for all practical purposes, non-blocking.
  void AcceptWaveform(BaseFloat samp_freq,
                      const VectorBase<BaseFloat> &wave_part);

  /// Returns the number of pieces of waveform that are still waiting to be
  /// processed.  This may be useful for calling code to judge whether to supply
  /// more waveform or to wait.
  int32 NumWaveformPiecesPending();

  /// You call this to inform the class that no more waveform will be provided;
  /// this allows it to flush out the last few frames of features, and is
  /// necessary if you want to call Wait() to wait until all decoding is done.
  /// After calling InputFinished() you cannot call AcceptWaveform any more.
  void InputFinished();

  /// You can call this if you don't want the decoding to proceed further with
  /// this utterance.  It just won't do any more processing, but you can still
  /// use the lattice from the decoding that it's already done.  Note: it may
  /// still continue decoding up to decode_batch_size (default: 2) frames of
  /// data before the decoding thread exits.  You can call Wait() after calling
  /// this, if you want to wait for that.
  void TerminateDecoding();

  /// This call will block until all the data has been decoded; it must only be
  /// called after either InputFinished() has been called or TerminateDecoding() has
  /// been called; otherwise, to call it is an error.
  void Wait();

  /// Finalizes the decoding. Cleans up and prunes remaining tokens, so the final
  /// lattice is faster to obtain.  May not be called unless either InputFinished()
  /// or TerminateDecoding() has been called.  If InputFinished() was called, it
  /// calls Wait() to ensure that the decoding has finished (it's not an error
  /// if you already called Wait()).
  void FinalizeDecoding();

  /// Returns *approximately* (ignoring end effects), the number of frames of
  /// data that we expect given the amount of data that the pipeline has
  /// received via AcceptWaveform().  (ignores small end effects).  This might
  /// be useful in application code to compare with NumFramesDecoded() and gauge
  /// how much latency there is.
  int32 NumFramesReceivedApprox() const;

  /// Returns the number of frames currently decoded.  Caution: don't rely on
  /// the lattice having exactly this number if you get it after this call, as
  /// it may increase after this-- unless you've already called either
  /// TerminateDecoding() or InputFinished(), followed by Wait().
  int32 NumFramesDecoded() const;

  /// Gets the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.  If this is at the end of the utterance,
  /// you might want to first call FinalizeDecoding() first; this will make this
  /// call return faster.
  /// If no frames have been decoded yet, it will set clat to a lattice with
  /// a single state that is final and with unit weight (no cost or alignment).
  /// The output to final_relative_cost (if non-NULL) is a number >= 0 that's
  /// closer to 0 if a final-state was close to the best-likelihood state
  /// active on the last frame, at the time we obtained the lattice.
  void GetLattice(bool end_of_utterance,
                  CompactLattice *clat,
                  BaseFloat *final_relative_cost) const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  /// If no frames have been decoded yet, it will set best_path to a lattice with
  /// a single state that is final and with unit weight (no cost).
  /// The output to final_relative_cost (if non-NULL) is a number >= 0 that's
  /// closer to 0 if a final-state were close to the best-likelihood state
  /// active on the last frame, at the time we got the best path.
  void GetBestPath(bool end_of_utterance,
                   Lattice *best_path,
                   BaseFloat *final_relative_cost) const;

  /// This function calls EndpointDetected from online-endpoint.h,
  /// with the required arguments.
  bool EndpointDetected(const OnlineEndpointConfig &config);

  /// Outputs the adaptation state of the feature pipeline to "adaptation_state".  This
  /// mostly stores stats for iVector estimation, and will generally be called at the
  /// end of an utterance, assuming it's a scenario where each speaker is seen for
  /// more than one utterance.
  /// You may only call this function after either calling TerminateDecoding() or
  /// InputFinished, and then Wait().  Otherwise it is an error.
  void GetAdaptationState(OnlineIvectorExtractorAdaptationState *adaptation_state);

  /// Outputs the OnlineCmvnState of the feature pipeline to "cmvn_stat".  This
  /// stores cmvn stats for the non-iVector features, and will be called at the
  /// end of an utterance, assuming it's a scenario where each speaker is seen for
  /// more than one utterance.
  /// You may only call this function after either calling TerminateDecoding() or
  /// InputFinished, and then Wait().  Otherwise it is an error.
  void GetCmvnState(OnlineCmvnState *cmvn_state);

  /// Gets the remaining, un-decoded part of the waveform and returns the sample
  /// rate.  May only be called after Wait(), and it only makes sense to call
  /// this if you called TerminateDecoding() before Wait().  The idea is that
  /// you can then provide this un-decoded piece of waveform to another decoder.
  BaseFloat GetRemainingWaveform(Vector<BaseFloat> *waveform_out) const;

  ~SingleUtteranceNnet2DecoderThreaded();
 private:

  // This function will instruct all threads to abort operation as soon as they
  // can safely do so, by calling SetAbort() in the threads
  void AbortAllThreads(bool error);

  // This function waits for all the threads that have been spawned. It is
  // called in the destructor and Wait(). If called twice it is not an error.
  void WaitForAllThreads();



  // this function runs the thread that does the feature extraction and
  // neural-net evaluation. In case of failure, calls
  // me->AbortAllThreads(true).
  static void RunNnetEvaluation(SingleUtteranceNnet2DecoderThreaded *me);
  // member-function version of RunNnetEvaluation, called by RunNnetEvaluation.
  bool RunNnetEvaluationInternal();
  // the following function is called inside RunNnetEvaluationInternal(); it
  // takes the log and subtracts the prior.
  void ProcessLoglikes(const CuVector<BaseFloat> &log_inv_prior,
                       CuMatrixBase<BaseFloat> *loglikes);
  // called from RunNnetEvaluationInternal().  Returns true in the normal case,
  // false on error; if it returns false, then we expect that the calling thread
  // will terminate.  This assumes the caller has already
  // locked feature_pipeline_mutex_.
  bool FeatureComputation(int32 num_frames_output);


  // this function runs the thread that does the neural-net evaluation.
  // In case of failure, calls me->AbortAllThreads(true).
  static void RunDecoderSearch(SingleUtteranceNnet2DecoderThreaded *me);
  // member-function version of RunDecoderSearch, called by RunDecoderSearch.
  bool RunDecoderSearchInternal();


  // Member variables:

  OnlineNnet2DecodingThreadedConfig config_;

  const nnet2::AmNnet &am_nnet_;

  const TransitionModel &tmodel_;


  // sampling_rate_ is set the first time AcceptWaveform is called.
  BaseFloat sampling_rate_;
  // A record of how many samples have been provided so
  // far via calls to AcceptWaveform.
  int64 num_samples_received_;

  // The next two variables are written to by AcceptWaveform from the main
  // thread, and read by the feature-processing thread; they are guarded by
  // waveform_synchronizer_.  There is no bound on the buffer size here.
  // Later-arriving data is appended to the vector.  When InputFinished() is
  // called from the main thread, the main thread sets input_finished_ = true.
  // sampling_rate_ is only needed for checking that it matches the config.
  bool input_finished_;
  std::deque< Vector<BaseFloat>* > input_waveform_;


  ThreadSynchronizer waveform_synchronizer_;

  // feature_pipeline_ is accessed by the nnet-evaluation thread, by the main
  // thread if GetAdaptionState() is called, and by the decoding thread via
  // ComputeCurrentTraceback() if online silence weighting is being used.  It is
  // guarded by feature_pipeline_mutex_.
  OnlineNnet2FeaturePipeline feature_pipeline_;
  std::mutex feature_pipeline_mutex_;

  // The next two variables are required only for implementation of the function
  // GetRemainingWaveform().  After we take waveform from the input_waveform_
  // queue to be processed into features, we put them onto this deque.  Then we
  // discard from this queue any that we can discard because we have already
  // decoded those frames (see num_frames_decoded_), and we increment
  // num_samples_discarded_ by the corresponding number of samples.
  std::deque< Vector<BaseFloat>* > processed_waveform_;
  int64 num_samples_discarded_;

  // This object is used to control the (optional) downweighting of silence in iVector estimation,
  // which is based on the decoder traceback.
  OnlineSilenceWeighting silence_weighting_;
  std::mutex silence_weighting_mutex_;


  // this Decodable object just stores a matrix of scaled log-likelihoods
  // obtained by the nnet-evaluation thread.  It is produced by the
  // nnet-evaluation thread and consumed by the decoder-search thread.  The
  // decoding thread sets num_frames_decoded_ so the nnet-evaluation thread
  // knows which frames of log-likelihoods it can discard.  Both of these
  // variables are guarded by decodable_synchronizer_.  Note:
  // the num_frames_decoded_ may be less than the current number of frames
  // the decoder has decoded; the decoder thread sets this variable when it
  // locks this mutex.
  DecodableMatrixMappedOffset decodable_;
  int32 num_frames_decoded_;
  ThreadSynchronizer decodable_synchronizer_;

  // the decoder_ object contains everything related to the graph search.
  LatticeFasterOnlineDecoder decoder_;
  // decoder_mutex_ guards the decoder_ object.  It is usually held by the decoding
  // thread (where it is released and re-obtained on each frame), but is obtained
  // by the main (parent) thread if you call functions like NumFramesDecoded(),
  // GetLattice() and GetBestPath().
  mutable std::mutex decoder_mutex_;  // declared as mutable because we mutate
                                      // this mutex in const methods

  // This contains the thread pointers for the nnet-evaluation and
  // decoder-search threads respectively (or NULL if they have been joined in
  // Wait()).
  std::thread threads_[2];

  // This is set to true if AbortAllThreads was called for any reason, including
  // if someone called TerminateDecoding().
  bool abort_;

  // This is set to true if any kind of unexpected error is encountered,
  // including if exceptions are raised in any of the threads.  Will normally
  // be a coding error, malloc failure-- something we should never encounter.
  bool error_;

};


/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_NNET2_DECODING_THREADED_H_
