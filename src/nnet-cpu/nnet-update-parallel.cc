// nnet/nnet-update-parallel.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-update-parallel.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

void ExamplesRepository::AcceptExamples(
    std::vector<NnetTrainingExample> *examples) {
  KALDI_ASSERT(!examples->empty());
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  examples_.swap(*examples);
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesDone() {
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  done_ = true;
  full_semaphore_.Signal();
}

bool ExamplesRepository::ProvideExamples(
    std::vector<NnetTrainingExample> *examples) {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return false; // no examples to return-- all finished.
  } else {
    KALDI_ASSERT(!examples_.empty() && examples->empty());
    examples->swap(examples_);
    empty_semaphore_.Signal();
    return true;
  }
}

class DoBackpropParallelClass: public MultiThreadable {
 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
  DoBackpropParallelClass(const Nnet &nnet,
                          ExamplesRepository *repository,
                          int64 *num_frames_ptr,
                          double *log_prob_ptr,
                          Nnet *nnet_to_update):
      nnet_(nnet), repository_(repository),
      nnet_to_update_(nnet_to_update),
      nnet_to_update_orig_(nnet_to_update),
      num_frames_ptr_(num_frames_ptr),
      log_prob_ptr_(log_prob_ptr),
      num_frames_(0),
      log_prob_(0.0) { }
  
  // The following constructor is called multiple times within
  // the RunMultiThreaded template function.
  DoBackpropParallelClass(const DoBackpropParallelClass &other):
      nnet_(other.nnet_),
      repository_(other.repository_),
      nnet_to_update_orig_(other.nnet_to_update_orig_),
      num_frames_ptr_(other.num_frames_ptr_),
      log_prob_ptr_(other.log_prob_ptr_),
      num_frames_(0),
      log_prob_(0.0) {
    if (other.nnet_to_update_ == &other.nnet_) { // Hogwild;
      // we update the original object directly.
      nnet_to_update_ = other.nnet_to_update_;
    } else {
      // Gradient accumulation; to ensure correctness, we
      // work on separate copies of the gradient object, which we'll
      // sum at the end.
      nnet_to_update_ = new Nnet(*other.nnet_to_update_);
      // our "nnet_to_update_" variable is a copy of the neural network
      // we are to update (presumably a gradient).  If we don't set these
      // to zero we would end up adding multiple copies of the any initial
      // gradient that "nnet_to_update_" contained when we initialize
      // the first instance of the class.
      nnet_to_update_->SetZero(true);      
    }    
  }
  // This does the main function of the class.
  void operator () () {
    KALDI_ASSERT(nnet_to_update_ != nnet_to_update_orig_ ||
                 nnet_to_update_ == &nnet_);
    std::vector<NnetTrainingExample> examples;
    while (repository_->ProvideExamples(&examples)) {
      // This is a function call to a function defined in
      // nnet-update.h
      BaseFloat tot_loglike = DoBackprop(nnet_, examples, nnet_to_update_);
      num_frames_ += examples.size();
      log_prob_ += tot_loglike;
      KALDI_VLOG(1) << "Thread " << thread_id_ << " saw "
                    << num_frames_ << " frames so far; likelihood per "
                    << " frame so far is " << (log_prob_ / num_frames_);
      examples.clear();
    }    
  }
  
  ~DoBackpropParallelClass() {
    if (nnet_to_update_orig_ != nnet_to_update_) {
      // This branch is only taken if this instance of the class is
      // one of the multiple instances allocated inside the RunMultiThreaded
      // template function, *and* we're doing gradient accumulation, summing
      // up the gradients of the individual processes.
      // In the hogwild case, there is nothing to do.
      int32 n = nnet_to_update_orig_->NumUpdatableComponents();
      Vector<BaseFloat> scales(n);
      scales.Set(1.0);
      nnet_to_update_orig_->AddNnet(scales, *nnet_to_update_);
      delete nnet_to_update_;
    }
    *log_prob_ptr_ += log_prob_;
    *num_frames_ptr_ += num_frames_;
  }
 private:
  const Nnet &nnet_;
  ExamplesRepository *repository_;
  Nnet *nnet_to_update_;
  Nnet *nnet_to_update_orig_;
  int64 *num_frames_ptr_;
  double *log_prob_ptr_;
  int64 num_frames_;
  double log_prob_; // log-like times num frames.
};

BaseFloat DoBackpropParallel(const Nnet &nnet,
                             int32 minibatch_size,
                             SequentialNnetTrainingExampleReader *examples_reader,
                             int64 *num_frames,
                             Nnet *nnet_to_update) {
  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *num_frames = 0;
  
  DoBackpropParallelClass c(nnet, &repository, num_frames,
                            &tot_log_prob, nnet_to_update);

  {
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<DoBackpropParallelClass> m(g_num_threads, c);
    
    std::vector<NnetTrainingExample> examples;
    for (; !examples_reader->Done(); examples_reader->Next()) {
      examples.push_back(examples_reader->Value());
      if (examples.size() == minibatch_size)
        repository.AcceptExamples(&examples);
    }
    if (!examples.empty()) // partial minibatch.
      repository.AcceptExamples(&examples);
    // Here, the destructor of "m" re-joins the threads, and
    // does the summing of the gradients if we're doing gradient
    // computation (i.e. &nnet != nnet_to_update).  This gets
    // done in the destructors of the objects of type
    // DoBackpropParallelClass.
    repository.ExamplesDone();
  }
  KALDI_LOG << "Did backprop on " << *num_frames << " examples, average log-prob "
            << "per frame is " << (tot_log_prob / *num_frames);
  return tot_log_prob;
}

  
} // namespace
