// online2/online-nnet2-decoding-threaded.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-nnet2-decoding-threaded.h"
#include "nnet2/nnet-compute-online.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

ThreadSynchronizer::ThreadSynchronizer():
    abort_(false), 
    producer_waiting_(false),
    consumer_waiting_(false),
    num_errors_(0) {
  producer_semaphore_.Signal();
  consumer_semaphore_.Signal();
}

bool ThreadSynchronizer::Lock(ThreadType t) {
  if (abort_)
    return false;
  if (t == ThreadSynchronizer::kProducer) {
    producer_semaphore_.Wait();
  } else {
    consumer_semaphore_.Wait();
  }
  if (abort_)
    return false;
  mutex_.Lock();
  held_by_ = t;
  if (abort_) {
    mutex_.Unlock();
    return false;
  } else {
    return true;
  }
}

bool ThreadSynchronizer::UnlockSuccess(ThreadType t) {
  if (t == ThreadSynchronizer::kProducer) {
    producer_semaphore_.Signal();  // next Lock won't wait.
    if (consumer_waiting_) {
      consumer_semaphore_.Signal();
      consumer_waiting_ = false;
    }
  } else {
    consumer_semaphore_.Signal(); // next Lock won't wait.
    if (producer_waiting_) {
      producer_semaphore_.Signal();
      producer_waiting_ = false;
    }
    
  }  
  mutex_.Unlock();
  return !abort_;
}

bool ThreadSynchronizer::UnlockFailure(ThreadType t) {

  KALDI_ASSERT(held_by_ == t && "Code error: unlocking a mutex you don't hold.");

  if (t == ThreadSynchronizer::kProducer) {
    KALDI_ASSERT(!producer_waiting_ && "code error.");
    producer_waiting_ = true;
  } else {
    KALDI_ASSERT(!consumer_waiting_ && "code error.");
    consumer_waiting_ = true;
  }
  mutex_.Unlock();
  return !abort_;
}

void ThreadSynchronizer::SetAbort() {
  abort_ = true;
  // we signal the semaphores just in case someone was waiting on either of
  // them.
  producer_semaphore_.Signal();
  consumer_semaphore_.Signal();
}

ThreadSynchronizer::~ThreadSynchronizer() {
}

// static
void OnlineNnet2DecodingThreadedConfig::Check() {
  KALDI_ASSERT(max_buffered_features > 1);
  KALDI_ASSERT(feature_batch_size > 0);
  KALDI_ASSERT(max_loglikes_copy >= 0);
  KALDI_ASSERT(nnet_batch_size > 0);
  KALDI_ASSERT(decode_batch_size >= 1);
}


SingleUtteranceNnet2DecoderThreaded::SingleUtteranceNnet2DecoderThreaded(
    const OnlineNnet2DecodingThreadedConfig &config,
    const TransitionModel &tmodel,
    const nnet2::AmNnet &am_nnet,
    const fst::Fst<fst::StdArc> &fst,
    const OnlineNnet2FeaturePipelineInfo &feature_info,
    const OnlineIvectorExtractorAdaptationState &adaptation_state):
    config_(config), am_nnet_(am_nnet), tmodel_(tmodel), sampling_rate_(0.0),
    num_samples_received_(0), input_finished_(false),
    feature_pipeline_(feature_info), feature_buffer_start_frame_(0),
    feature_buffer_finished_(false), decodable_(tmodel),
    num_frames_decoded_(0), decoder_(fst, config_.decoder_opts),
    abort_(false), error_(false) {
  // if the user supplies an adaptation state that was not freshly initialized,
  // it means that we take the adaptation state from the previous
  // utterance(s)... this only makes sense if theose previous utterance(s) are
  // believed to be from the same speaker.
  feature_pipeline_.SetAdaptationState(adaptation_state);
  // spawn threads.

  pthread_attr_t pthread_attr;
  pthread_attr_init(&pthread_attr);
  int32 ret;

  // Note: if the constructor throws an exception, the corresponding destructor
  // will not be called.  So we don't have to be careful about setting the
  // thread pointers to NULL after we've joined them.
  if ((ret=pthread_create(&(threads_[0]),
                          &pthread_attr, RunFeatureExtraction,
                          (void*)this)) != 0) {
    const char *c = strerror(ret);
    if (c == NULL) { c = "[NULL]"; }
    KALDI_ERR << "Error creating thread, errno was: " << c;
  }
  if ((ret=pthread_create(&(threads_[1]),
                          &pthread_attr, RunNnetEvaluation,
                          (void*)this)) != 0) {
    const char *c = strerror(ret);
    if (c == NULL) { c = "[NULL]"; }
    bool error = true;
    AbortAllThreads(error);
    KALDI_WARN << "Error creating thread, errno was: " << c
               << " (will rejoin already-created threads).";
    if (pthread_join(threads_[0], NULL)) {
      KALDI_ERR << "Error rejoining thread.";
    } else {
      KALDI_ERR << "Error creating thread, errno was: " << c;
    }
  }
  if ((ret=pthread_create(&(threads_[2]),
                          &pthread_attr, RunDecoderSearch,
                          (void*)this)) != 0) {
    const char *c = strerror(ret);
    if (c == NULL) { c = "[NULL]"; }
    bool error = true;
    AbortAllThreads(error);
    KALDI_WARN << "Error creating thread, errno was: " << c
               << " (will rejoin already-created threads).";
    if (pthread_join(threads_[0], NULL) || pthread_join(threads_[1], NULL)) {
      KALDI_ERR << "Error rejoining thread.";
    } else {
      KALDI_ERR << "Error creating thread, errno was: " << c;
    }
  }
}


SingleUtteranceNnet2DecoderThreaded::~SingleUtteranceNnet2DecoderThreaded() {
  if (!abort_) {
    // If we have not already started the process of aborting the threads, do so now.
    bool error = false;
    AbortAllThreads(error);
  }
  // join all the threads (this avoids leaving zombie threads around, or threads
  // that might be accessing deconstructed object).
  WaitForAllThreads();
  DeletePointers(&input_waveform_);
  DeletePointers(&feature_buffer_);
}

void SingleUtteranceNnet2DecoderThreaded::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &wave_part) {
  if (sampling_rate_ <= 0.0)
    sampling_rate_ = sampling_rate;
  else {
    KALDI_ASSERT(sampling_rate == sampling_rate_);
  }
  num_samples_received_ += wave_part.Dim();
  
  if (wave_part.Dim() == 0) return;
  if (!waveform_synchronizer_.Lock(ThreadSynchronizer::kProducer)) {
    KALDI_ERR << "Failure locking mutex: decoding aborted.";
  }

  Vector<BaseFloat> *new_part = new Vector<BaseFloat>(wave_part);
  input_waveform_.push_back(new_part);
  // we always unlock with success because there is no buffer size limitation
  // for the waveform so no reason why we might wait.
  waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer);
}

int32 SingleUtteranceNnet2DecoderThreaded::NumFramesReceivedApprox() const {
  return num_samples_received_ /
      (sampling_rate_ * feature_pipeline_.FrameShiftInSeconds());
}

void SingleUtteranceNnet2DecoderThreaded::InputFinished() {
  // setting input_finished_ = true informs the feature-processing pipeline
  // to expect no more input, and to flush out the last few frames if there
  // is any latency in the pipeline (e.g. due to pitch).
  if (!waveform_synchronizer_.Lock(ThreadSynchronizer::kProducer)) {
    KALDI_ERR << "Failure locking mutex: decoding aborted.";
  }
  KALDI_ASSERT(!input_finished_ && "InputFinished called twice");
  input_finished_ = true;
  waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer);
}

void SingleUtteranceNnet2DecoderThreaded::TerminateDecoding() {
  bool error = false;
  AbortAllThreads(error);
}

void SingleUtteranceNnet2DecoderThreaded::Wait() {
  if (!input_finished_ && !abort_) {
    KALDI_ERR << "You cannot call Wait() before calling either InputFinished() "
              << "or TerminateDecoding().";
  }
  WaitForAllThreads();
}

void SingleUtteranceNnet2DecoderThreaded::FinalizeDecoding() {
  if (KALDI_PTHREAD_PTR(threads_[0]) != 0) {
    KALDI_ERR << "It is an error to call FinalizeDecoding before Wait().";
  }
  decoder_.FinalizeDecoding();
}


void SingleUtteranceNnet2DecoderThreaded::GetAdaptationState(
    OnlineIvectorExtractorAdaptationState *adaptation_state) {
  feature_pipeline_mutex_.Lock();  // If this blocks, it shouldn't be for very long.
  feature_pipeline_.GetAdaptationState(adaptation_state);
  feature_pipeline_mutex_.Unlock();  // If this blocks, it won't be for very long.  
}
  
void SingleUtteranceNnet2DecoderThreaded::GetLattice(
    bool end_of_utterance,
    CompactLattice *clat,
    BaseFloat *final_relative_cost) const {
  clat->DeleteStates();
  // we'll make an exception to the normal const rules, for mutexes, since
  // we're not really changing the class.
  const_cast<Mutex&>(decoder_mutex_).Lock();
  if (final_relative_cost != NULL)
    *final_relative_cost = decoder_.FinalRelativeCost();
  if (decoder_.NumFramesDecoded() == 0) {
    const_cast<Mutex&>(decoder_mutex_).Unlock();    
    clat->SetFinal(clat->AddState(),
                   CompactLatticeWeight::One());
    return;
  }
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);
  const_cast<Mutex&>(decoder_mutex_).Unlock();
  
  if (!config_.decoder_opts.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = config_.decoder_opts.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      tmodel_, &raw_lat, lat_beam, clat, config_.decoder_opts.det_opts);
}

void SingleUtteranceNnet2DecoderThreaded::GetBestPath(
    bool end_of_utterance,
    Lattice *best_path,
    BaseFloat *final_relative_cost) const {
  // we'll make an exception to the normal const rules, for mutexes, since
  // we're not really changing the class.
  const_cast<Mutex&>(decoder_mutex_).Lock();
  if (decoder_.NumFramesDecoded() == 0) {
    // It's possible that this if-statement is not necessary because we'd get this
    // anyway if we just called GetBestPath on the decoder.
    best_path->DeleteStates();
    best_path->SetFinal(best_path->AddState(),
                        LatticeWeight::One());
    if (final_relative_cost != NULL)    
      *final_relative_cost = std::numeric_limits<BaseFloat>::infinity();
  } else {
    decoder_.GetBestPath(best_path,
                         end_of_utterance);
    if (final_relative_cost != NULL)
      *final_relative_cost = decoder_.FinalRelativeCost();
  }
  const_cast<Mutex&>(decoder_mutex_).Unlock();
}

void SingleUtteranceNnet2DecoderThreaded::AbortAllThreads(bool error) {
  abort_ = true;
  if (error)
    error_ = true;
  waveform_synchronizer_.SetAbort();
  feature_synchronizer_.SetAbort();
  decodable_synchronizer_.SetAbort();
}

int32 SingleUtteranceNnet2DecoderThreaded::NumFramesDecoded() const {
  const_cast<Mutex&>(decoder_mutex_).Lock();
  int32 ans =  decoder_.NumFramesDecoded();
  const_cast<Mutex&>(decoder_mutex_).Unlock();
  return ans;
}

void* SingleUtteranceNnet2DecoderThreaded::RunFeatureExtraction(void *ptr_in) {
  SingleUtteranceNnet2DecoderThreaded *me =
      reinterpret_cast<SingleUtteranceNnet2DecoderThreaded*>(ptr_in);
  try {
    if (!me->RunFeatureExtractionInternal() && !me->abort_)
      KALDI_ERR << "Returned abnormally and abort was not called";
  } catch(const std::exception &e) {
    KALDI_WARN << "Caught exception: " << e.what();
    // if an error happened in one thread, we need to make sure the other
    // threads can exit too.
    bool error = true;
    me->AbortAllThreads(error);
  }
  return NULL;
}

void* SingleUtteranceNnet2DecoderThreaded::RunNnetEvaluation(void *ptr_in) {
  SingleUtteranceNnet2DecoderThreaded *me =
      reinterpret_cast<SingleUtteranceNnet2DecoderThreaded*>(ptr_in);
  try {
    if (!me->RunNnetEvaluationInternal() && !me->abort_)
      KALDI_ERR << "Returned abnormally and abort was not called";
  } catch(const std::exception &e) {
    KALDI_WARN << "Caught exception: " << e.what();
    // if an error happened in one thread, we need to make sure the other
    // threads can exit too.
    bool error = true;
    me->AbortAllThreads(error);
  }
  return NULL;
}

void* SingleUtteranceNnet2DecoderThreaded::RunDecoderSearch(void *ptr_in) {
  SingleUtteranceNnet2DecoderThreaded *me =
      reinterpret_cast<SingleUtteranceNnet2DecoderThreaded*>(ptr_in);
  try {
    if (!me->RunDecoderSearchInternal() && !me->abort_)
      KALDI_ERR << "Returned abnormally and abort was not called";
  } catch(const std::exception &e) {
    KALDI_WARN << "Caught exception: " << e.what();
    // if an error happened in one thread, we need to make sure the other threads can exit too.
    bool error = true;
    me->AbortAllThreads(error);
  }
  return NULL;
}


void SingleUtteranceNnet2DecoderThreaded::WaitForAllThreads() {
  for (int32 i = 0; i < 3; i++) {  // there are 3 spawned threads.
    pthread_t &thread = threads_[i];
    if (KALDI_PTHREAD_PTR(thread) != 0) {
      if (pthread_join(thread, NULL)) {
        KALDI_ERR << "Error rejoining thread";  // this should not happen.
      }
      KALDI_PTHREAD_PTR(thread) = 0;
    }
  }
  if (error_) {
    KALDI_ERR << "Error encountered during decoding.  See above.";
  }
}

bool SingleUtteranceNnet2DecoderThreaded::RunFeatureExtractionInternal() {
  // Note: if any of the functions Lock, UnlockSuccess, UnlockFailure return
  // false, it is because AbortAllThreads() called, and we return false
  // immediately.

  // num_frames_output is a local variable that keeps track of how many
  // frames we have output to the feature buffer, for this utterance.
  int32 num_frames_output = 0;
  
  while (true) {
    // First deal with accepting input.
    if (!waveform_synchronizer_.Lock(ThreadSynchronizer::kConsumer))
      return false;
    if (input_waveform_.empty()) {
      if (input_finished_ &&
          !feature_pipeline_.IsLastFrame(feature_pipeline_.NumFramesReady()-1)) {
        // the main thread called InputFinished() and set input_finished_, and
        // we haven't yet registered that fact.  This is progress so
        // UnlockSuccess().
        feature_pipeline_.InputFinished();
        if (!waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
          return false;
      } else {
        // there was no input to process.  However, we only call UnlockFailure() if we
        // are blocked on the fact that there is no input to process; otherwise we
        // call UnlockSuccess().
        if (num_frames_output == feature_pipeline_.NumFramesReady()) {
          // we need to wait until there is more input.
          if (!waveform_synchronizer_.UnlockFailure(ThreadSynchronizer::kConsumer))
            return false;
        } else { // we can keep looping.
          if (!waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
            return false;
        }
      }
    } else {  // there is more wav data.
      { // braces clarify scope of locking.
        feature_pipeline_mutex_.Lock();
        for (size_t i = 0; i < input_waveform_.size(); i++)
          if (input_waveform_[i]->Dim() != 0)
            feature_pipeline_.AcceptWaveform(sampling_rate_, *input_waveform_[i]);
        feature_pipeline_mutex_.Unlock();
      }
      DeletePointers(&input_waveform_);      
      input_waveform_.clear();
      if (!waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
        return false;
    }

    if (!feature_synchronizer_.Lock(ThreadSynchronizer::kProducer)) return false;
    
    if (feature_buffer_.size() >= config_.max_buffered_features) {
      // we need to block on the output buffer.
      if (!feature_synchronizer_.UnlockFailure(ThreadSynchronizer::kProducer))
        return false;
    } else {

      { // braces clarify scope of locking.      
        feature_pipeline_mutex_.Lock();
        // There is buffer space available; deal with producing output.
        int32 cur_size = feature_buffer_.size(),
            batch_size = config_.feature_batch_size,
            feat_dim = feature_pipeline_.Dim();
        
        for (int32 t = feature_buffer_start_frame_ +
                 static_cast<int32>(feature_buffer_.size());
                t < feature_buffer_start_frame_ + config_.max_buffered_features &&
                 t < feature_buffer_start_frame_ + cur_size + batch_size &&
                 t < feature_pipeline_.NumFramesReady(); t++) {
          Vector<BaseFloat> *feats = new Vector<BaseFloat>(feat_dim, kUndefined);
          // Note: most of the actual computation occurs.
          feature_pipeline_.GetFrame(t, feats);
          feature_buffer_.push_back(feats);
        }
        num_frames_output = feature_buffer_start_frame_ + feature_buffer_.size();
        if (feature_pipeline_.IsLastFrame(num_frames_output - 1)) {
          // e.g. user called InputFinished() and we already saw the last frame.
          feature_buffer_finished_ = true;
        }
        feature_pipeline_mutex_.Unlock();
      }      
      if (!feature_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer))
        return false;
        
      if (feature_buffer_finished_) return true;
    }
  }
}

bool SingleUtteranceNnet2DecoderThreaded::RunNnetEvaluationInternal() {
  // if any of the Lock/Unlock functions return false, it's because AbortAllThreads()
  // was called.
  
  // This object is responsible for keeping track of the context, and avoiding
  // re-computing things we've already computed.
  bool pad_input = true;
  nnet2::NnetOnlineComputer computer(am_nnet_.GetNnet(), pad_input);

  // we declare the following as CuVector just to enable GPU support, but
  // we expect this code to be run on CPU in the normal case.
  CuVector<BaseFloat> log_inv_priors(am_nnet_.Priors());
  log_inv_priors.ApplyFloor(1.0e-20);  // should have no effect.
  log_inv_priors.ApplyLog();
  log_inv_priors.Scale(-1.0);
  
  int32 num_frames_output = 0;
  
  while (true) {
    bool last_time = false;
  
    if (!feature_synchronizer_.Lock(ThreadSynchronizer::kConsumer))
      return false;

    CuMatrix<BaseFloat> cu_loglikes;
    
    if (feature_buffer_.empty()) {
      if (feature_buffer_finished_) {
        // flush out the last few frames.  Note: this is the only place from
        // which we check feature_buffer_finished_, and we'll exit the loop, so
        // if we reach here it must be the first time it was true.
        last_time = true;
        if (!feature_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
          return false;
        computer.Flush(&cu_loglikes);
      } else {
        // there is nothing to do because there is no input.  Next call to Lock
        // should block till the feature-processing thread does something.
        if (!feature_synchronizer_.UnlockFailure(ThreadSynchronizer::kConsumer))
          return false;
      }
    } else {
      int32 num_frames_evaluate = std::min<int32>(feature_buffer_.size(),
                                                  config_.nnet_batch_size),
          feat_dim = feature_buffer_[0]->Dim();
      Matrix<BaseFloat> feats(num_frames_evaluate, feat_dim);
      for (int32 i = 0; i < num_frames_evaluate; i++) {
        feats.Row(i).CopyFromVec(*(feature_buffer_[i]));
        delete feature_buffer_[i];
      }
      feature_buffer_.erase(feature_buffer_.begin(),
                            feature_buffer_.begin() + num_frames_evaluate);
      feature_buffer_start_frame_ += num_frames_evaluate;
      if (!feature_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
        return false;

      CuMatrix<BaseFloat> cu_feats;
      cu_feats.Swap(&feats);  // If we don't have a GPU (and not having a GPU is
                              // the normal expected use-case for this code),
                              // this would be a lightweight operation, swapping
                              // pointers.

      KALDI_VLOG(4) << "Computing chunk of " << cu_feats.NumRows() << " frames "
                    << "of nnet.";
      computer.Compute(cu_feats, &cu_loglikes);
      cu_loglikes.ApplyFloor(1.0e-20);
      cu_loglikes.ApplyLog();
      // take the log-posteriors and turn them into pseudo-log-likelihoods by
      // dividing by the pdf priors; then scale by the acoustic scale.
      if (cu_loglikes.NumRows() != 0) {
        cu_loglikes.AddVecToRows(1.0, log_inv_priors);
        cu_loglikes.Scale(config_.acoustic_scale);
      }
    }

    Matrix<BaseFloat> loglikes;
    loglikes.Swap(&cu_loglikes);  // If we don't have a GPU (and not having a
                                  // GPU is the normal expected use-case for
                                  // this code), this would be a lightweight
                                  // operation, swapping pointers.


    // OK, at this point we may have some newly created log-likes and we want to
    // give them to the decoding thread.  
    
    int32 num_loglike_frames = loglikes.NumRows();

    if (loglikes.NumRows() != 0) {  // if we need to output some loglikes...
      while (true) {
        // we may have to grab and release the decodable mutex
        // a few times before it's ready to accept the loglikes.
        if (!decodable_synchronizer_.Lock(ThreadSynchronizer::kProducer))
          return false;
        int32 num_frames_decoded = num_frames_decoded_;
        // we can't have output fewer frames than were decoded.
        KALDI_ASSERT(num_frames_output >= num_frames_decoded);
        if (num_frames_output - num_frames_decoded < config_.max_loglikes_copy) {
          // If we would have to copy fewer than config_.max_loglikes_copy
          // previously evaluated log-likelihoods inside the decodable object..
          int32 frames_to_discard = num_frames_decoded_ -
              decodable_.FirstAvailableFrame();
          KALDI_ASSERT(frames_to_discard >= 0);
          num_frames_output += num_loglike_frames;
          decodable_.AcceptLoglikes(&loglikes, frames_to_discard);
          if (!decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer))
            return false;
          break;  // break from the innermost while loop.
        } else {
          // we want the next call to Lock to block until the decoder has
          //  processed more frames.
          if (!decodable_synchronizer_.UnlockFailure(ThreadSynchronizer::kProducer))
            return false;
        }
      }
    }
    if (last_time) {
      // Inform the decodable object that there will be no more input.
      if (!decodable_synchronizer_.Lock(ThreadSynchronizer::kProducer))
        return false;
      decodable_.InputIsFinished();
      if (!decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer))
        return false;
      return true;
    }
  }
}
  

bool SingleUtteranceNnet2DecoderThreaded::RunDecoderSearchInternal() {
  int32 num_frames_decoded = 0;  // this is just a copy of decoder_->NumFramesDecoded();
  decoder_.InitDecoding();
  while (true) {  // decode at most one frame each loop.
    if (!decodable_synchronizer_.Lock(ThreadSynchronizer::kConsumer))
      return false; // AbortAllThreads() called.
    if (decodable_.NumFramesReady() <= num_frames_decoded) {
      // no frames available to decode.
      KALDI_ASSERT(decodable_.NumFramesReady() == num_frames_decoded);
      if (decodable_.IsLastFrame(num_frames_decoded - 1)) {
        decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer);
        return true;  // exit from this thread; we're done.
      } else {
        // we were not able to advance the decoding due to no available
        // input.  The next call will ensure that the next call to
        // decodable_synchronizer_.Lock() will wait.
        if (!decodable_synchronizer_.UnlockFailure(ThreadSynchronizer::kConsumer))
          return false;
      }
    } else {
      // Decode at most config_.decode_batch_size frames (e.g. 1 or 2).
      decoder_mutex_.Lock();
      decoder_.AdvanceDecoding(&decodable_, config_.decode_batch_size);
      num_frames_decoded = decoder_.NumFramesDecoded();
      decoder_mutex_.Unlock();
      num_frames_decoded_ = num_frames_decoded;
      if (!decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
        return false;
    }
  }
}

bool SingleUtteranceNnet2DecoderThreaded::EndpointDetected(
    const OnlineEndpointConfig &config) {
  decoder_mutex_.Lock();
  bool ans = kaldi::EndpointDetected(config, tmodel_,
                                     feature_pipeline_.FrameShiftInSeconds(),
                                     decoder_);
  decoder_mutex_.Unlock();
  return ans;
}



}  // namespace kaldi

