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
#include "util/kaldi-thread.h"

namespace kaldi {

ThreadSynchronizer::ThreadSynchronizer():
    abort_(false),
    producer_waiting_(false),
    consumer_waiting_(false) {
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
  mutex_.lock();
  held_by_ = t;
  if (abort_) {
    mutex_.unlock();
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
  mutex_.unlock();
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
  mutex_.unlock();
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
    const OnlineIvectorExtractorAdaptationState &adaptation_state,
    const OnlineCmvnState &cmvn_state):
  config_(config), am_nnet_(am_nnet), tmodel_(tmodel), sampling_rate_(0.0),
  num_samples_received_(0), input_finished_(false),
  feature_pipeline_(feature_info),
  num_samples_discarded_(0),
  silence_weighting_(tmodel, feature_info.silence_weighting_config),
  decodable_(tmodel),
  num_frames_decoded_(0), decoder_(fst, config_.decoder_opts),
  abort_(false), error_(false) {
  // if the user supplies an adaptation state that was not freshly initialized,
  // it means that we take the adaptation state from the previous
  // utterance(s)... this only makes sense if theose previous utterance(s) are
  // believed to be from the same speaker.
  feature_pipeline_.SetAdaptationState(adaptation_state);
  feature_pipeline_.SetCmvnState(cmvn_state);
  // spawn threads.
  threads_[0] = std::thread(RunNnetEvaluation, this);
  decoder_.InitDecoding();
  threads_[1] = std::thread(RunDecoderSearch, this);
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
  while (!input_waveform_.empty()) {
    delete input_waveform_.front();
    input_waveform_.pop_front();
  }
  while (!processed_waveform_.empty()) {
    delete processed_waveform_.front();
    processed_waveform_.pop_front();
  }
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

int32 SingleUtteranceNnet2DecoderThreaded::NumWaveformPiecesPending() {
  // Note RE locking: what we really want here is just to lock the mutex.  As a
  // side effect, because of the way the synchronizer code works, it will also
  // increment the semaphore and might wake up the consumer thread.  This will
  // possibly make it do a little useless work (go around a loop once), but
  // won't really do any harm.  Perhaps we should have implemented a version of
  // the Lock function that takes no arguments.
  if (!waveform_synchronizer_.Lock(ThreadSynchronizer::kProducer)) {
    KALDI_ERR << "Failure locking mutex: decoding aborted.";
  }
  int32 ans = input_waveform_.size();
  waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer);
  return ans;
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
  if (threads_[0].joinable()) {
    KALDI_ERR << "It is an error to call FinalizeDecoding before Wait().";
  }
  decoder_.FinalizeDecoding();
}

BaseFloat SingleUtteranceNnet2DecoderThreaded::GetRemainingWaveform(
    Vector<BaseFloat> *waveform) const {
  if (threads_[0].joinable()) {
    KALDI_ERR << "It is an error to call GetRemainingWaveform before Wait().";
  }
  int64 num_samples_stored = 0;  // number of samples we still have.
  std::vector< Vector<BaseFloat>* > all_pieces;
  std::deque< Vector<BaseFloat>* >::const_iterator iter;
  for (iter = processed_waveform_.begin(); iter != processed_waveform_.end();
       ++iter) {
    num_samples_stored += (*iter)->Dim();
    all_pieces.push_back(*iter);
  }
  for (iter = input_waveform_.begin(); iter != input_waveform_.end(); ++iter) {
    num_samples_stored += (*iter)->Dim();
    all_pieces.push_back(*iter);
  }
  int64 samples_shift_per_frame =
      sampling_rate_ * feature_pipeline_.FrameShiftInSeconds();
  int64 num_samples_to_discard = samples_shift_per_frame * num_frames_decoded_;
  KALDI_ASSERT(num_samples_to_discard >= num_samples_discarded_);

  // num_samp_discard is how many samples we must discard from our stored
  // samples.
  int64 num_samp_discard = num_samples_to_discard - num_samples_discarded_,
      num_samp_keep = num_samples_stored - num_samp_discard;
  KALDI_ASSERT(num_samp_discard <= num_samples_stored && num_samp_keep >= 0);
  waveform->Resize(num_samp_keep, kUndefined);
  int32 offset = 0; // offset in output waveform.  assume output waveform is no
                    // larger than int32.
  for (size_t i = 0; i < all_pieces.size(); i++) {
    Vector<BaseFloat> *this_piece = all_pieces[i];
    int32 this_dim = this_piece->Dim();
    if (num_samp_discard >= this_dim) {
      num_samp_discard -= this_dim;
    } else {
      // normal case is num_samp_discard = 0.
      int32 this_dim_keep = this_dim - num_samp_discard;
      waveform->Range(offset, this_dim_keep).CopyFromVec(
          this_piece->Range(num_samp_discard, this_dim_keep));
      offset += this_dim_keep;
      num_samp_discard = 0;
    }
  }
  KALDI_ASSERT(offset == num_samp_keep && num_samp_discard == 0);
  return sampling_rate_;
}

void SingleUtteranceNnet2DecoderThreaded::GetAdaptationState(
    OnlineIvectorExtractorAdaptationState *adaptation_state) {
  std::lock_guard<std::mutex> lock(feature_pipeline_mutex_);
  // If this blocks, it shouldn't be for very long.
  feature_pipeline_.GetAdaptationState(adaptation_state);
}

void SingleUtteranceNnet2DecoderThreaded::GetCmvnState(
    OnlineCmvnState *cmvn_state) {
  std::lock_guard<std::mutex> lock(feature_pipeline_mutex_);
  // If this blocks, it shouldn't be for very long.
  feature_pipeline_.GetCmvnState(cmvn_state);
}

void SingleUtteranceNnet2DecoderThreaded::GetLattice(
    bool end_of_utterance,
    CompactLattice *clat,
    BaseFloat *final_relative_cost) const {
  clat->DeleteStates();
  decoder_mutex_.lock();
  if (final_relative_cost != NULL)
    *final_relative_cost = decoder_.FinalRelativeCost();
  if (decoder_.NumFramesDecoded() == 0) {
    decoder_mutex_.unlock();
    clat->SetFinal(clat->AddState(),
                   CompactLatticeWeight::One());
    return;
  }
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);
  decoder_mutex_.unlock();

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
  std::lock_guard<std::mutex> lock(decoder_mutex_);
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
}

void SingleUtteranceNnet2DecoderThreaded::AbortAllThreads(bool error) {
  abort_ = true;
  if (error)
    error_ = true;
  waveform_synchronizer_.SetAbort();
  decodable_synchronizer_.SetAbort();
}

int32 SingleUtteranceNnet2DecoderThreaded::NumFramesDecoded() const {
  std::lock_guard<std::mutex> lock(decoder_mutex_);
  return decoder_.NumFramesDecoded();
}

void SingleUtteranceNnet2DecoderThreaded::RunNnetEvaluation(
    SingleUtteranceNnet2DecoderThreaded *me) {
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
}

void SingleUtteranceNnet2DecoderThreaded::RunDecoderSearch(
    SingleUtteranceNnet2DecoderThreaded *me) {
  try {
    if (!me->RunDecoderSearchInternal() && !me->abort_)
      KALDI_ERR << "Returned abnormally and abort was not called";
  } catch(const std::exception &e) {
    KALDI_WARN << "Caught exception: " << e.what();
    // if an error happened in one thread, we need to make sure the other threads can exit too.
    bool error = true;
    me->AbortAllThreads(error);
  }
}


void SingleUtteranceNnet2DecoderThreaded::WaitForAllThreads() {
  for (int32 i = 0; i < 2; i++) {  // there are 2 spawned threads.
    if (threads_[i].joinable())
      threads_[i].join();
  }
  if (error_)
    KALDI_ERR << "Error encountered during decoding.  See above.";
}


void SingleUtteranceNnet2DecoderThreaded::ProcessLoglikes(
    const CuVector<BaseFloat> &log_inv_prior,
    CuMatrixBase<BaseFloat> *cu_loglikes) {
  if (cu_loglikes->NumRows() != 0) {
    cu_loglikes->ApplyFloor(1.0e-20);
    cu_loglikes->ApplyLog();
    // take the log-posteriors and turn them into pseudo-log-likelihoods by
    // dividing by the pdf priors; then scale by the acoustic scale.
    cu_loglikes->AddVecToRows(1.0, log_inv_prior);
    cu_loglikes->Scale(config_.acoustic_scale);
  }
}

// called from RunNnetEvaluationInternal().  Returns true in the normal case,
// false on error; if it returns false, then we expect that the calling thread
// will terminate.  This assumes the calling thread has already
// locked feature_pipeline_mutex_.
bool SingleUtteranceNnet2DecoderThreaded::FeatureComputation(
    int32 num_frames_consumed) {

  int32 num_frames_ready = feature_pipeline_.NumFramesReady(),
      num_frames_usable = num_frames_ready - num_frames_consumed;
  bool features_done = feature_pipeline_.IsLastFrame(num_frames_ready - 1);
  KALDI_ASSERT(num_frames_usable >= 0);
  if (features_done) {
    return true;  // nothing to do. (but not an error).
  } else {
    if (num_frames_usable >= config_.nnet_batch_size)
      return true;  // We don't need more data yet.

    // Now try to get more data, if we can.
    if (!waveform_synchronizer_.Lock(ThreadSynchronizer::kConsumer)) {
      return false;
    }
    // we've got the lock.
    if (input_waveform_.empty()) {  // we got no data
      if (input_finished_ &&
          !feature_pipeline_.IsLastFrame(feature_pipeline_.NumFramesReady()-1)) {
        // the main thread called InputFinished() and set input_finished_, and
        // we haven't yet registered that fact.  This is progress so
        // unlock with UnlockSuccess().
        feature_pipeline_.InputFinished();
        return waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer);
      } else {
        // there is no progress.  Unlock with UnlockFailure() so the next call to
        // waveform_synchronizer_.Lock() will lock.
        return waveform_synchronizer_.UnlockFailure(ThreadSynchronizer::kConsumer);
      }
    } else {  // we got some data.  Only take enough of the waveform to
              // give us a maximum nnet batch size of frames to decode.
      while (num_frames_usable < config_.nnet_batch_size &&
             !input_waveform_.empty()) {
        feature_pipeline_.AcceptWaveform(sampling_rate_, *input_waveform_.front());
        processed_waveform_.push_back(input_waveform_.front());
        input_waveform_.pop_front();
        num_frames_ready = feature_pipeline_.NumFramesReady();
        num_frames_usable = num_frames_ready - num_frames_consumed;
      }
      // Delete already-processed pieces of waveform if we have already decoded
      // those frames.  (If not already decoded, we keep them around for the
      // sake of GetRemainingWaveform()).
      int32 samples_shift_per_frame =
          sampling_rate_ * feature_pipeline_.FrameShiftInSeconds();
      while (!processed_waveform_.empty() &&
             num_samples_discarded_ + processed_waveform_.front()->Dim() <
             samples_shift_per_frame * num_frames_decoded_) {
        num_samples_discarded_ += processed_waveform_.front()->Dim();
        delete processed_waveform_.front();
        processed_waveform_.pop_front();
      }
      return waveform_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer);
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
  CuVector<BaseFloat> log_inv_prior(am_nnet_.Priors());
  log_inv_prior.ApplyFloor(1.0e-20);  // should have no effect.
  log_inv_prior.ApplyLog();
  log_inv_prior.Scale(-1.0);

  // we'll have num_frames_consumed >= num_frames_output; num_frames_consumed is
  // the number of feature frames consumed by the nnet computation,
  // num_frames_output is the number of frames of loglikes the nnet computation
  // has produced, which may be less than num_frames_consumed due to the
  // right-context of the network.
  int32 num_frames_consumed = 0, num_frames_output = 0;

  while (true) {
    bool last_time = false;

    /****** Begin locking of feature pipeline mutex. ******/
    feature_pipeline_mutex_.lock();
    if (!FeatureComputation(num_frames_consumed)) {  // error
      feature_pipeline_mutex_.unlock();
      return false;
    }
    // take care of silence weighting.
    if (silence_weighting_.Active() &&
        feature_pipeline_.IvectorFeature() != NULL) {
      silence_weighting_mutex_.lock();
      std::vector<std::pair<int32, BaseFloat> > delta_weights;
      silence_weighting_.GetDeltaWeights(
          feature_pipeline_.IvectorFeature()->NumFramesReady(),
          &delta_weights);
      silence_weighting_mutex_.unlock();
      feature_pipeline_.IvectorFeature()->UpdateFrameWeights(delta_weights);
    }

    int32 num_frames_ready = feature_pipeline_.NumFramesReady(),
        num_frames_usable = num_frames_ready - num_frames_consumed;
    bool features_done = feature_pipeline_.IsLastFrame(num_frames_ready - 1);

    int32 num_frames_evaluate = std::min<int32>(num_frames_usable,
                                                config_.nnet_batch_size);

    Matrix<BaseFloat> feats;
    if (num_frames_evaluate > 0) {
      // we have something to do...
      feats.Resize(num_frames_evaluate, feature_pipeline_.Dim());
      for (int32 i = 0; i < num_frames_evaluate; i++) {
        int32 t = num_frames_consumed + i;
        SubVector<BaseFloat> feat(feats, i);
        feature_pipeline_.GetFrame(t, &feat);
      }
    }
    /****** End locking of feature pipeline mutex. ******/
    feature_pipeline_mutex_.unlock();

    CuMatrix<BaseFloat> cu_loglikes;

    if (feats.NumRows() == 0) {
      if (features_done) {
        // flush out the last few frames.  Note: this is the only place from
        // which we check feature_buffer_finished_, and we'll exit the loop, so
        // if we reach here it must be the first time it was true.
        last_time = true;
        computer.Flush(&cu_loglikes);
        ProcessLoglikes(log_inv_prior, &cu_loglikes);
      }
    } else {
      CuMatrix<BaseFloat> cu_feats;
      cu_feats.Swap(&feats);  // If we don't have a GPU (and not having a GPU is
                              // the normal expected use-case for this code),
                              // this would be a lightweight operation, swapping
                              // pointers.

      computer.Compute(cu_feats, &cu_loglikes);
      num_frames_consumed += cu_feats.NumRows();
      ProcessLoglikes(log_inv_prior, &cu_loglikes);
    }

    Matrix<BaseFloat> loglikes;
    loglikes.Swap(&cu_loglikes);  // If we don't have a GPU (and not having a
                                  // GPU is the normal expected use-case for
                                  // this code), this would be a lightweight
                                  // operation, swapping pointers.


    // OK, at this point we may have some newly created log-likes and we want to
    // give them to the decoding thread.

    int32 num_loglike_frames = loglikes.NumRows();

    if (num_loglike_frames != 0) {  // if we need to output some loglikes...
      while (true) {
        // we may have to grab and release the decodable mutex
        // a few times before it's ready to accept the loglikes.
        if (!decodable_synchronizer_.Lock(ThreadSynchronizer::kProducer))
          return false;
        int32 num_frames_decoded = num_frames_decoded_;
        // we can't have output fewer frames than were decoded.
        KALDI_ASSERT(num_frames_output >= num_frames_decoded);
        if (num_frames_output - num_frames_decoded <= config_.max_loglikes_copy) {
          // If we would have to copy fewer than config_.max_loglikes_copy
          // previously output log-likelihoods inside the decodable object, then
          // we go ahead and copy them to that object.
          int32 frames_to_discard = num_frames_decoded_ -
              decodable_.FirstAvailableFrame();
          KALDI_ASSERT(frames_to_discard >= 0);
          num_frames_output += num_loglike_frames;
          decodable_.AcceptLoglikes(&loglikes, frames_to_discard);
          if (!decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kProducer))
            return false;
          break;  // break from the innermost while loop.
        } else {
          // There are too many frames already available to the decoder, that it
          // hasn't processed yet, and we don't want them to have to be copied
          // inside AcceptLoglikes(), so we wait for a bit.
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
      KALDI_ASSERT(num_frames_consumed == num_frames_output);
      return true;
    }
  }
}


bool SingleUtteranceNnet2DecoderThreaded::RunDecoderSearchInternal() {
  int32 num_frames_decoded = 0;  // this is just a copy of decoder_->NumFramesDecoded();
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
      decoder_mutex_.lock();
      decoder_.AdvanceDecoding(&decodable_, config_.decode_batch_size);
      num_frames_decoded = decoder_.NumFramesDecoded();
      if (silence_weighting_.Active()) {
        std::lock_guard<std::mutex> lock(silence_weighting_mutex_);
        // the next function does not trace back all the way; it's very fast.
        silence_weighting_.ComputeCurrentTraceback(decoder_);
      }
      decoder_mutex_.unlock();
      num_frames_decoded_ = num_frames_decoded;
      if (!decodable_synchronizer_.UnlockSuccess(ThreadSynchronizer::kConsumer))
        return false;
    }
  }
}

bool SingleUtteranceNnet2DecoderThreaded::EndpointDetected(
    const OnlineEndpointConfig &config) {
  std::lock_guard<std::mutex> lock(decoder_mutex_);
  return kaldi::EndpointDetected(config, tmodel_,
                                 feature_pipeline_.FrameShiftInSeconds(),
                                 decoder_);
}



}  // namespace kaldi
