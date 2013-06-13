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
#include "thread/kaldi-mutex.h"

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
                          Nnet *nnet_to_update,
                          bool store_separate_gradients):
      nnet_(nnet), repository_(repository),
      nnet_to_update_(nnet_to_update),
      nnet_to_update_orig_(nnet_to_update),
      store_separate_gradients_(store_separate_gradients),
      num_frames_ptr_(num_frames_ptr),
      log_prob_ptr_(log_prob_ptr),
      num_frames_(0),
      log_prob_(0.0) { }
  
  // The following constructor is called multiple times within
  // the RunMultiThreaded template function.
  DoBackpropParallelClass(const DoBackpropParallelClass &other):
      nnet_(other.nnet_),
      repository_(other.repository_),
      nnet_to_update_(other.nnet_to_update_),
      nnet_to_update_orig_(other.nnet_to_update_orig_),
      store_separate_gradients_(other.store_separate_gradients_),
      num_frames_ptr_(other.num_frames_ptr_),
      log_prob_ptr_(other.log_prob_ptr_),
      num_frames_(0),
      log_prob_(0.0) {
    if (store_separate_gradients_) {
      // To ensure correctness, we work on separate copies of the gradient
      // object, which we'll sum at the end.  This is used for exact gradient
      // computation.
      nnet_to_update_ = new Nnet(*(other.nnet_to_update_));
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
      // template function, *and* store_separate_gradients_ has been set to true.
      // In the typical hogwild case, we don't do this.
      nnet_to_update_orig_->AddNnet(1.0, *nnet_to_update_);
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
  bool store_separate_gradients_;
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

  // This function assumes you want the exact gradient, if
  // nnet_to_update != &nnet.
  const bool store_separate_gradients = (nnet_to_update != &nnet);
  
  DoBackpropParallelClass c(nnet, &repository, num_frames,
                            &tot_log_prob, nnet_to_update,
                            store_separate_gradients);

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

class ApplyMomentumClass: public MultiThreadable {
 public:
  ApplyMomentumClass(Nnet *gradient,
                     Nnet *nnet,
                     BaseFloat momentum_minibatches,
                     Semaphore *minibatch_semaphore,
                     Mutex *done_mutex,
                     bool *done):
      gradient_(gradient), nnet_(nnet),
      momentum_minibatches_(momentum_minibatches),
      minibatch_semaphore_(minibatch_semaphore),
      done_mutex_(done_mutex),
      done_(done) {
    KALDI_ASSERT(momentum_minibatches_ > 1.0);
  }
  
  // Use the default copy constructor.

  void operator () () {
    
    while (true) {
      minibatch_semaphore_->Wait();

      // "to_transfer" is what proportion of the gradient to transfer to the model.
      BaseFloat to_transfer = 1.0 / momentum_minibatches_,
          remaining = 1.0 - to_transfer;
      
      while (minibatch_semaphore_->TryWait()) { // If the semaphore has been signaled multiple
        // times before we were able to process it, let's handle it all in one big batch,
        // to avoid this thread getting behind.
        to_transfer += remaining / momentum_minibatches_;
        remaining = 1.0 - to_transfer;
      }
      // Note: the following two lines of code that implement momentum are not
      // 100% ideal.  The issue is that if the gradient gets modified between
      // the following two commands, the associated terms in the gradient will
      // get under-counted.  But this effect becomes very small, in relative
      // terms, as "momentum_minibatches" becomes large, and we anticipate that
      // "momentum_minibatches" will be >> 1 (e.g. 10, or 100, or 1000), so we
      // don't worry about this too much.
      nnet_->AddNnet(to_transfer, gradient_, remaining);
      
      done_mutex_->Lock();
      if (*done_) {
        done_mutex_->Unlock();
        return; // we're done.
      } else {
        done_mutex_->Unlock();
      }
    }
  }

  ~ApplyMomentumClass() { }
 private:
  Nnet *gradient_;
  Nnet *nnet_;
  BaseFloat momentum_minibatches_;
  Semaphore *minibatch_semaphore_;
  Mutex *done_mutex_;
  bool *done_;
};
  


BaseFloat DoBackpropParallelMomentum(
    int32 minibatch_size,
    BaseFloat momentum_minibatches,
    SequentialNnetTrainingExampleReader *examples_reader,
    int64 *num_frames,
    Nnet *nnet) {

  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *num_frames = 0;

  KALDI_ASSERT(momentum_minibatches > 1.0 &&
               "Bad value for --momentum-minibatches: <= 1.0");
  
  Nnet nnet_gradient(*nnet);
  // although we use this to represent the gradient we tell it not to
  // treat it as the gradient because we want the preconditioning
  // turned on.
  const bool treat_as_gradient = false;
  nnet_gradient.SetZero(treat_as_gradient);
  
  const bool store_separate_gradients = false;
  
  DoBackpropParallelClass c(*nnet, &repository, num_frames,
                            &tot_log_prob, &nnet_gradient,
                            store_separate_gradients);

  {
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<DoBackpropParallelClass> m(g_num_threads, c);

  
    Mutex done_mutex;
    bool done = false;
    Semaphore minibatch_semaphore;
    ApplyMomentumClass momentum_class(&nnet_gradient, nnet,
                                      momentum_minibatches,
                                      &minibatch_semaphore,
                                      &done_mutex, &done);
    MultiThreader<ApplyMomentumClass> n(1, momentum_class); // Spawn one thread
    // to handle the momentum; it transfers from the "nnet_gradient" to
    // "nnet".
    
    std::vector<NnetTrainingExample> examples;
    for (; !examples_reader->Done(); examples_reader->Next()) {
      examples.push_back(examples_reader->Value());
      if (examples.size() == minibatch_size) {
        repository.AcceptExamples(&examples);
        minibatch_semaphore.Signal();
      }
    }

    // Let the momentum thread know that we're done.
    done_mutex.Lock();
    done = true;
    minibatch_semaphore.Signal();
    done_mutex.Unlock();
    
    if (!examples.empty()) // partial minibatch.
      repository.AcceptExamples(&examples);

    // Here, the destructor of "m" re-joins the threads, and
    // does the summing of the gradients if we're doing gradient
    // computation (i.e. &nnet != nnet_to_update).  This gets
    // done in the destructors of the objects of type
    // DoBackpropParallelMomentumClass.
    repository.ExamplesDone();
  }
  
  // Add as much of the accumulated gradient as we can without incurring
  // any more instability than we have already incurred.  I choose to set this
  // as at most, the equivalent of processing two times "g_num_threads"
  // minibatches.   This is just to deal with end effects in a sensible way,
  // without running much extra risk of instability; it's not very critical.
  BaseFloat heuristic = 2.0;
  BaseFloat coeff = std::min(static_cast<BaseFloat>(1.0),
                             heuristic * g_num_threads / momentum_minibatches);
  nnet->AddNnet(coeff, nnet_gradient);

  KALDI_LOG << "Did backprop on " << *num_frames << " examples with momentum "
            << " time constant equal to " << momentum_minibatches
            << " minibatches, average " << "log-prob per frame is "
            << (tot_log_prob / *num_frames);

  return tot_log_prob;
}


BaseFloat DoBackpropParallel(const Nnet &nnet,
                             int32 minibatch_size,
                             int32 num_threads,
                             const std::vector<NnetTrainingExample> &egs,
                             int64 *num_frames,
                             Nnet *nnet_to_update) {
  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *num_frames = 0;
  const bool store_separate_gradients = (nnet_to_update != &nnet);
  
  DoBackpropParallelClass c(nnet, &repository, num_frames,
                            &tot_log_prob, nnet_to_update,
                            store_separate_gradients);

  {
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<DoBackpropParallelClass> m(num_threads, c);
    
    int32 num_egs = egs.size();
    for (int32 offset = 0; offset < num_egs; offset += minibatch_size) {
      int32 this_minibatch_size = std::min(minibatch_size, num_egs - offset);

      // We waste a little time copying the examples here, but it's very minor.
      std::vector<NnetTrainingExample> examples(egs.begin() + offset,
                                                egs.begin() + offset + this_minibatch_size);
    
      repository.AcceptExamples(&examples);
    }
    
    // Here, the destructor of "m" re-joins the threads, and
    // does the summing of the gradients if we're doing gradient
    // computation (i.e. &nnet != nnet_to_update).  This gets
    // done in the destructors of the objects of type
    // DoBackpropParallelClass.
    repository.ExamplesDone();
  }
  KALDI_VLOG(2) << "Did backprop on " << *num_frames << " examples, average log-prob "
                << "per frame is " << (tot_log_prob / *num_frames);
  return tot_log_prob;
}

  
} // namespace
