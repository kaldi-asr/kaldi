// nnet2/nnet-update-parallel.cc

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

#include "nnet2/nnet-update-parallel.h"
#include "nnet2/nnet-update.h"
#include "thread/kaldi-thread.h"
#include "thread/kaldi-mutex.h"
#include <numeric>

namespace kaldi {
namespace nnet2 {

/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples,
  /// with a batch of examples.  [It will empty the vector "examples").
  void AcceptExamples(std::vector<NnetExample> *examples);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples.
  void ExamplesDone();
  
  /// This function is called by the code that does the training.  It gets the
  /// training examples, and if they are available, puts them in "examples" and
  /// returns true.  It returns false when there are no examples left and
  /// ExamplesDone() has been called.
  bool ProvideExamples(std::vector<NnetExample> *examples);
  
  ExamplesRepository(): empty_semaphore_(1), done_(false) { }
 private:
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;

  std::vector<NnetExample> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};


void ExamplesRepository::AcceptExamples(
    std::vector<NnetExample> *examples) {
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
    std::vector<NnetExample> *examples) {
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
                          double *tot_weight_ptr,
                          double *log_prob_ptr,
                          Nnet *nnet_to_update,
                          bool store_separate_gradients):
      nnet_(nnet), repository_(repository),
      nnet_to_update_(nnet_to_update),
      nnet_to_update_orig_(nnet_to_update),
      store_separate_gradients_(store_separate_gradients),
      tot_weight_ptr_(tot_weight_ptr),
      log_prob_ptr_(log_prob_ptr),
      tot_weight_(0.0),
      log_prob_(0.0) { }
  
  // The following constructor is called multiple times within
  // the RunMultiThreaded template function.
  DoBackpropParallelClass(const DoBackpropParallelClass &other):
      nnet_(other.nnet_),
      repository_(other.repository_),
      nnet_to_update_(other.nnet_to_update_),
      nnet_to_update_orig_(other.nnet_to_update_orig_),
      store_separate_gradients_(other.store_separate_gradients_),
      tot_weight_ptr_(other.tot_weight_ptr_),
      log_prob_ptr_(other.log_prob_ptr_),
      tot_weight_(0),
      log_prob_(0.0) {
    if (store_separate_gradients_) {
      // To ensure correctness, we work on separate copies of the gradient
      // object, which we'll sum at the end.  This is used for exact gradient
      // computation.
      if (other.nnet_to_update_ != NULL) {
        nnet_to_update_ = new Nnet(*(other.nnet_to_update_));
        // our "nnet_to_update_" variable is a copy of the neural network
        // we are to update (presumably a gradient).  If we don't set these
        // to zero we would end up adding multiple copies of the any initial
        // gradient that "nnet_to_update_" contained when we initialize
        // the first instance of the class.
        nnet_to_update_->SetZero(true);      
      } else { // support case where we don't really need a gradient.
        nnet_to_update_ = NULL;
      }
    }    
  }
  // This does the main function of the class.
  void operator () () {
    std::vector<NnetExample> examples;
    while (repository_->ProvideExamples(&examples)) {
      // This is a function call to a function defined in
      // nnet-update.h
      double tot_loglike;
      if (nnet_to_update_ != NULL) 
        tot_loglike = DoBackprop(nnet_, examples, nnet_to_update_);
      else
        tot_loglike = ComputeNnetObjf(nnet_, examples);
      tot_weight_ += TotalNnetTrainingWeight(examples);
      log_prob_ += tot_loglike;
      KALDI_VLOG(4) << "Thread " << thread_id_ << " saw "
                    << tot_weight_ << " frames so far (weighted); likelihood "
                    << "per frame so far is " << (log_prob_ / tot_weight_);
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
    *tot_weight_ptr_ += tot_weight_;
  }
 private:
  const Nnet &nnet_;
  ExamplesRepository *repository_;
  Nnet *nnet_to_update_;
  Nnet *nnet_to_update_orig_;
  bool store_separate_gradients_;
  double *tot_weight_ptr_;
  double *log_prob_ptr_;
  double tot_weight_;
  double log_prob_; // log-like times num frames.
};


#if HAVE_CUDA == 1
double DoBackpropSingleThreaded(const Nnet &nnet,
                                int32 minibatch_size,
                                SequentialNnetExampleReader *examples_reader,
                                double *tot_weight_out,
                                Nnet *nnet_to_update) {
  double ans = 0.0, tot_weight = 0.0;
  KALDI_ASSERT(minibatch_size > 0);
  while (!examples_reader->Done()) {
    std::vector<NnetExample> egs;
    egs.reserve(minibatch_size);
    while (egs.size() < minibatch_size && examples_reader->Done()) {
      egs.push_back(examples_reader->Value());
      examples_reader->Next();
    }
    ans += DoBackprop(nnet, egs, nnet_to_update);
    tot_weight += TotalNnetTrainingWeight(egs);
  }
  *tot_weight_out = tot_weight;
  return ans;
}
#endif


double DoBackpropParallel(const Nnet &nnet,
                          int32 minibatch_size,
                          SequentialNnetExampleReader *examples_reader,
                          double *tot_weight,
                          Nnet *nnet_to_update) {
#if HAVE_CUDA == 1
  // Our GPU code won't work with multithreading; we do this
  // to enable it to work with this code in the single-threaded
  // case.
  if (CuDevice::Instantiate().Enabled())
    return DoBackpropSingleThreaded(nnet, minibatch_size, examples_reader,
                                    tot_weight, nnet_to_update);
#endif
  
  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *tot_weight = 0.0;

  // This function assumes you want the exact gradient, if
  // nnet_to_update != &nnet.
  const bool store_separate_gradients = (nnet_to_update != &nnet);
  
  DoBackpropParallelClass c(nnet, &repository, tot_weight,
                            &tot_log_prob, nnet_to_update,
                            store_separate_gradients);

  {
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<DoBackpropParallelClass> m(g_num_threads, c);
    
    std::vector<NnetExample> examples;
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
  KALDI_LOG << "Did backprop on " << *tot_weight << " examples, average log-prob "
            << "per frame is " << (tot_log_prob / *tot_weight);
  return tot_log_prob;
}


double DoBackpropSingleThreaded(const Nnet &nnet,
                                int32 minibatch_size,
                                const std::vector<NnetExample> &egs,
                                double *tot_weight,
                                Nnet *nnet_to_update) {
  double ans = 0.0;
  *tot_weight = TotalNnetTrainingWeight(egs);
  for (size_t i = 0; i < egs.size(); i += minibatch_size) {
    std::vector<NnetExample>::const_iterator end_iter =
      (i + minibatch_size > egs.size() ? egs.end() : 
       egs.begin() + i + minibatch_size);
    std::vector<NnetExample> this_egs(egs.begin() + i,
                                              end_iter);
    ans += DoBackprop(nnet, this_egs, nnet_to_update);
  }
  return ans;
}


double DoBackpropParallel(const Nnet &nnet,
                          int32 minibatch_size,
                          int32 num_threads,
                          const std::vector<NnetExample> &egs,
                          double *tot_weight,
                          Nnet *nnet_to_update) {
  if (num_threads == 1) // support GPUs: special case for 1 thread.
    return DoBackpropSingleThreaded(nnet, minibatch_size, egs, 
                                    tot_weight, nnet_to_update);

  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *tot_weight = 0;
  const bool store_separate_gradients = (nnet_to_update != &nnet);
  
  DoBackpropParallelClass c(nnet, &repository, tot_weight,
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
      std::vector<NnetExample> examples(egs.begin() + offset,
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
  KALDI_VLOG(2) << "Did backprop on " << *tot_weight << " examples, average log-prob "
                << "per frame is " << (tot_log_prob / *tot_weight);
  return tot_log_prob;
}

  
} // namespace nnet2
} // namespace kaldi
