// nnet2/train-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/train-nnet.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
namespace nnet2 {


class NnetExampleBackgroundReader {
 public:
  NnetExampleBackgroundReader(int32 minibatch_size,
                              Nnet *nnet,
                              SequentialNnetExampleReader *reader):
      minibatch_size_(minibatch_size), nnet_(nnet), reader_(reader),
      finished_(false) {
    // When this class is created, it spawns a thread which calls ReadExamples()
    // in the background.
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);
    int32 ret;
    // below, Run is the static class-member function.
    if ((ret=pthread_create(&thread_, &pthread_attr,
                            Run, static_cast<void*>(this)))) {
      const char *c = strerror(ret);
      if (c == NULL) { c = "[NULL]"; }
      KALDI_ERR << "Error creating thread, errno was: " << c;
    }
    // the following call is a signal that no-one is currently using the examples_ and
    // formatted_examples_ class members.
    consumer_semaphore_.Signal();
  }
  ~NnetExampleBackgroundReader() {
    if (KALDI_PTHREAD_PTR(thread_) == 0)
      KALDI_ERR << "No thread to join.";
    if (pthread_join(thread_, NULL))
      KALDI_ERR << "Error rejoining thread.";
  }

  // This will be called in a background thread.  It's responsible for
  // reading and formatting the examples.
  void ReadExamples() {
    KALDI_ASSERT(minibatch_size_ > 0);
    int32 minibatch_size = minibatch_size_;

    
    // Loop over minibatches...
    while (true) {
      // When the following call succeeds we interpret it as a signal that
      // we are free to write to the class-member variables examples_ and formatted_examples_.
      consumer_semaphore_.Wait();
      
      examples_.clear();
      examples_.reserve(minibatch_size);
      // Read the examples.
      for (; examples_.size() < minibatch_size && !reader_->Done(); reader_->Next())
        examples_.push_back(reader_->Value());

      // Format the examples as a single matrix.  The reason we do this here is
      // that it's a somewhat CPU-intensive operation (involves decompressing
      // the matrix), so we do it in a separate thread from the one that's
      // controlling the GPU (assuming we're using a GPU), so we can get better
      // GPU utilization.  If we have no GPU this doesn't hurt us.
      if (examples_.empty()) {
        formatted_examples_.Resize(0, 0);
        total_weight_ = 0.0;
      } else {
        FormatNnetInput(*nnet_, examples_, &formatted_examples_);
        total_weight_ = TotalNnetTrainingWeight(examples_);
      }
      
      bool finished = examples_.empty();
      
      // The following call alerts the main program thread (that calls
      // GetNextMinibatch() that it can how use the contents of
      // examples_ and formatted_examples_.
      producer_semaphore_.Signal();
      
      // If we just read an empty minibatch (because no more examples),
      // then return.
      if (finished)
        return;
    }
  }

  // this wrapper can be passed to pthread_create.
  static void* Run(void *ptr_in) {
    NnetExampleBackgroundReader *ptr =
        reinterpret_cast<NnetExampleBackgroundReader*>(ptr_in);
    ptr->ReadExamples();
    return NULL;
  }

  // This call makes available the next minibatch of input.  It returns
  // true if it got some, and false if there was no more available.
  // It is an error if you call this function after it has returned false.
  bool GetNextMinibatch(std::vector<NnetExample> *examples,
                        Matrix<BaseFloat> *formatted_examples,
                        double *total_weight) {
    KALDI_ASSERT(!finished_);
    // wait until examples_ and formatted_examples_ have been created by
    // the background thread.
    producer_semaphore_.Wait();
    // the calls to swap and Swap are lightweight.
    examples_.swap(*examples);
    formatted_examples_.Swap(formatted_examples);
    *total_weight = total_weight_;
    
    // signal the background thread that it is now free to write
    // again to examples_ and formatted_examples_.
    consumer_semaphore_.Signal();
    
    if (examples->empty()) {
      finished_ = true;
      return false;
    } else {
      return true;
    }
  }
  
 private:
  int32 minibatch_size_;
  Nnet *nnet_;
  SequentialNnetExampleReader *reader_;
  pthread_t thread_;
  
  std::vector<NnetExample> examples_;
  Matrix<BaseFloat> formatted_examples_;
  double total_weight_;  // total weight, from TotalNnetTrainingWeight(examples_).
                         // better to compute this in the background thread.
  
  Semaphore producer_semaphore_;
  Semaphore consumer_semaphore_;

  bool finished_;
};



int64 TrainNnetSimple(const NnetSimpleTrainerConfig &config,
                      Nnet *nnet,
                      SequentialNnetExampleReader *reader,
                      double *tot_weight_ptr,
                      double *tot_logprob_ptr) {
  int64 num_egs_processed = 0;
  double tot_weight = 0.0, tot_logprob = 0.0;
  NnetExampleBackgroundReader background_reader(config.minibatch_size,
                                                nnet, reader);
  KALDI_ASSERT(config.minibatches_per_phase > 0);
  while (true) {
    // Iterate over phases.  A phase of training is just a certain number of
    // minibatches, and its only significance is that it's the periodicity with
    // which we print diagnostics.
    double tot_weight_this_phase = 0.0, tot_logprob_this_phase = 0.0;

    int32 i;
    for (i = 0; i < config.minibatches_per_phase; i++) {
      std::vector<NnetExample> examples;
      Matrix<BaseFloat> examples_formatted;
      double minibatch_total_weight;  // this will normally equal minibatch size.
      if (!background_reader.GetNextMinibatch(&examples, &examples_formatted,
                                              &minibatch_total_weight))
        break;
      tot_logprob_this_phase += DoBackprop(*nnet, examples, &examples_formatted,
                                           nnet, NULL);
      tot_weight_this_phase += minibatch_total_weight;
      num_egs_processed += examples.size();
    }
    if (i != 0) {
      KALDI_LOG << "Training objective function (this phase) is "
                << (tot_logprob_this_phase / tot_weight_this_phase) << " over "
                << tot_weight_this_phase << " frames.";
    }
    tot_weight += tot_weight_this_phase;
    tot_logprob += tot_logprob_this_phase;
    if (i != config.minibatches_per_phase) {
      // did not get all the minibatches we wanted because no more input.
      // this is true if and only if we did "break" in the loop over i above.
      break;
    }
  }
  if (tot_weight == 0.0) {
    KALDI_WARN << "No data seen.";
  } else {
    KALDI_LOG << "Did backprop on " << tot_weight
              << " examples, average log-prob per frame is "
              << (tot_logprob / tot_weight);
    KALDI_LOG << "[this line is to be parsed by a script:] log-prob-per-frame="
              << (tot_logprob / tot_weight);
  }
  if (tot_weight_ptr) *tot_weight_ptr = tot_weight;
  if (tot_logprob_ptr) *tot_logprob_ptr = tot_logprob;
  return num_egs_processed;
}



} // namespace nnet2
} // namespace kaldi
