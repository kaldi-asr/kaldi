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
    std::vector<NnetTrainingExample> examples;
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
                                SequentialNnetTrainingExampleReader *examples_reader,
                                double *tot_weight_out,
                                Nnet *nnet_to_update) {
  double ans = 0.0, tot_weight = 0.0;
  KALDI_ASSERT(minibatch_size > 0);
  while (!examples_reader->Done()) {
    std::vector<NnetTrainingExample> egs;
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
                          SequentialNnetTrainingExampleReader *examples_reader,
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
  KALDI_LOG << "Did backprop on " << *tot_weight << " examples, average log-prob "
            << "per frame is " << (tot_log_prob / *tot_weight);
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
      // The following call will add "to_transfer * gradient_" to "nnet_",
      // and multiply "gradient_" by "remaining".
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
    double *tot_weight,
    Nnet *nnet) {

  KALDI_ASSERT(minibatch_size > 0);
  ExamplesRepository repository; // handles parallel programming issues regarding
  // the "examples" of data.
  double tot_log_prob = 0.0;
  *tot_weight = 0;

  KALDI_ASSERT(momentum_minibatches > 1.0 &&
               "Bad value for --momentum-minibatches: <= 1.0");
  
  Nnet nnet_gradient(*nnet);
  // although we use this to represent the gradient we tell it not to
  // treat it as the gradient because we want the preconditioning
  // turned on.
  const bool treat_as_gradient = false;
  nnet_gradient.SetZero(treat_as_gradient);
  
  const bool store_separate_gradients = false;
  
  DoBackpropParallelClass c(*nnet, &repository, tot_weight,
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
    int64 counter = 0;
    for (; !examples_reader->Done(); examples_reader->Next()) {
      counter++;
      if (counter % 10000 == 0) {
        KALDI_VLOG(2) << nnet->Info();
      }
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

  KALDI_LOG << "Did backprop on " << *tot_weight << " examples with momentum "
            << " time constant equal to " << momentum_minibatches
            << " minibatches, average " << "log-prob per frame is "
            << (tot_log_prob / *tot_weight);

  return tot_log_prob;
}

/// For the safe-backprop code: we will create just one object of type
/// SafeBackpropClass, and its operator () will run in the background.  This
/// object is responsible for periodically checking the objective function
/// change since the start of training (it estimates this change using
/// derivatives), and adjusting learning rates and model parameters to control
/// this if the change becomes too negative.  This object periodically gets
/// given some examples (we let num-threads minibatches be processed normally,
/// then use one for this purpose), and after it's done with them it "gives
/// them back" to the regular training code so they are not wasted.
/// We'll use the MultiThreadable interface as the base-class,
/// but this is just responsible for spawning a single thread of it in the
/// background, there's nothing more complex going on.

class SafeBackpropClass: public MultiThreadable {
 public:
  SafeBackpropClass(
      const SafeBackpropConfig &config,
      ExamplesRepository *safe_backprop_repository, // repository used just by this class
      ExamplesRepository *backprop_repository, // repository used by training code, where we put the examples when done.
      int32 backprop_minibatch_size, // minibatch size used by the basic training code
      Nnet *nnet,
      const int64 *counter):
      safe_config_(config),
      nnet_(nnet),
      counter_(counter),
      safe_backprop_repository_(safe_backprop_repository),
      backprop_repository_(backprop_repository),
      backprop_minibatch_size_(backprop_minibatch_size) { }
  
  // Use the default copy constructor.

  void operator () () {
    Nnet initial_nnet(*nnet_); // Make a copy of the neural net with its value at the
    // start of training.   
    
    std::vector<NnetTrainingExample> egs;

    int32 counter = 0;
    while (safe_backprop_repository_->ProvideExamples(&egs)) {
      KALDI_ASSERT(egs.size() == safe_config_.samples_per_test);
      // Use this batch of examples to check and possibly modify the per-layer
      // learning rates.
      CheckLearningRates(initial_nnet, egs, counter++);
      
      // Now we're done with the examples, give them back to
      // "backprop_repository" so they can be trained on.
      while (!egs.empty()) {
        std::vector<NnetTrainingExample> minibatch_egs;
        for (int32 i = 0; i < backprop_minibatch_size_ && !egs.empty(); i++) {
          minibatch_egs.push_back(egs.back());
          egs.pop_back();
        }
        backprop_repository_->AcceptExamples(&minibatch_egs);
      }
    }
  }

  /// This function computes a value "floor" such that if c(i) = max(floor,
  /// a(i)), we have that sum(c) = b.  At input, sum(a) must be <= b.  The value
  /// "floor" is chosen to ensure that this property holds.
  static BaseFloat GetFloorValue(const VectorBase<BaseFloat> &a,
                                 BaseFloat b) {
    BaseFloat cur_sum = a.Sum();
    if (cur_sum > b) {
      KALDI_WARN << "GetScaleFactors given invalid input.";
      return -1.0e+10;
    }
    std::vector<BaseFloat> v(a.Dim());
    for (int32 i = 0; i < a.Dim(); i++)
      v[i] = a(i);
    std::sort(v.begin(), v.end());
    // Now v is sorted in ascending order, from most negative to most positive.

    for (int32 i = 0; i+1 < a.Dim(); i++) {
      // For each i, we try a "floor" value such that
      // v[i] <= floor <= v[i+1], and see if this would
      // suffice.
      BaseFloat remainder_sum = std::accumulate(v.begin() + i + 1, v.end(), 0.0);
      // remainder_sum is the sum of all elements past i.
      BaseFloat remaining_value = b - remainder_sum;
      
      BaseFloat floor_val = remaining_value / (i + 1); // If we chose this floor value,
      // we'd have the property that we want.
      if (floor_val < v[i]) // code error.
        KALDI_ERR << "Floor-val " << floor_val << " < v[i] = " << v[i];
      if (floor_val <= v[i+1]) return floor_val; // It's within the range
      // we wanted.
    }
    // If we fell off the loop, then floor_val will be above all the values.
    BaseFloat floor_val = b / a.Dim();
    if (floor_val < v[a.Dim() - 1]) // code error.
      KALDI_ERR << "Unexpected floor value " << floor_val << " < "
                << v[a.Dim()-1];
    return floor_val;
  }

  void GetLearningRates(const Nnet &initial_nnet,
                        const VectorBase<BaseFloat> &scales,
                        int32 counter,
                        VectorBase<BaseFloat> *learning_rates,
                        VectorBase<BaseFloat> *modified_scales) {
    int32 nu = initial_nnet.NumUpdatableComponents();
    KALDI_ASSERT(scales.Dim() == nu && learning_rates->Dim() == nu
                 && modified_scales->Dim() == nu);
    Vector<BaseFloat> initial_learning_rates(nu),
        current_learning_rates(nu), current_scales(nu);
    initial_nnet.GetLearningRates(&initial_learning_rates);
    nnet_->GetLearningRates(&current_learning_rates);
    
    
    current_scales.AddVecDivVec(1.0, current_learning_rates,
                                initial_learning_rates, 0.0);

    KALDI_ASSERT(safe_config_.relax_factor >= 1.0);
    
    // "current_scales" are the current scaling factors on the learning rates.
    for (int32 i = 0; i < scales.Dim(); i++) {
      BaseFloat scale; // scale relative to current-scale on the learning rate.
      if (scales(i) < 1.0) scale = scales(i);
      else if (current_scales(i) < 1.0) {
        scale = safe_config_.relax_factor;
        if (1.0 / current_scales(i) < scale)
          scale = 1.0 / current_scales(i);
      } else {
        scale = 1.0;
      }
      // At this point, scale is typically equal to scales(i) which is <= 1.0,
      // but it may be larger, up to safe_config_.relax_factor (default: 1.1),
      // if we have previously scaled down these learning rates but don't
      // currently want to scale them down.
      (*learning_rates)(i) = current_learning_rates(i) * scale;

      scale = pow(scale, 1.0 / sqrt(counter + 1.0));
      // Here, counter would be 0 on the first call to this function and 1 on
      // the second.  The point of this call is to make the learning rates
      // change more slowly, the more times we have checked them.
      
      (*modified_scales)(i) = scale;

      
    }
  }
  
  void ModifyLearningRates(const Nnet &initial_nnet,
                           BaseFloat max_degradation,
                           const VectorBase<BaseFloat> &objf_change_per_layer,
                           int32 counter) {
    int32 nu = initial_nnet.NumUpdatableComponents();    
    BaseFloat tot_change = objf_change_per_layer.Sum();

    /*
      We apply a scaling factor to each layer, to both the parameters and the
      learning rates, in order to limit this change.  The assumption will be
      that the degradation is due to parameter noise, and it's reasonable to
      assume that the objf change per layer is proportional to the square of
      the scaling factor that we apply (i.e. we started out close to the optimum).

      We may formulate the problem of obtaining the scaling factors, as follows.
      First get the target objf changes as follows.
      Let the objf changes per layer be o(i); most of these will be negative.
      Define p(i) = max(o(i), floor),
      where floor [a negative number] is the most negative allowed objf change.
      Then compute "floor" such that sum_i p(i) = max_degradation.

      Once we have this then we can compute the desired scaling factors (on the
      learning rates and parameter-changes) as
        s(i) = sqrt(p(i) / o(i)).
    */
    
    Vector<BaseFloat> modified_objf_change(objf_change_per_layer);

    if (tot_change < -max_degradation) {
      BaseFloat floor = GetFloorValue(modified_objf_change, -max_degradation);
      modified_objf_change.ApplyFloor(floor);
      if (fabs(modified_objf_change.Sum() - (-max_degradation)) > 0.01) {
        KALDI_ERR << "Objf change has wrong sum (code error) " << modified_objf_change.Sum()
                  << " != " << -max_degradation;
      }
    }
    Vector<BaseFloat> scales(modified_objf_change.Dim());
    
    for (int32 i = 0; i < modified_objf_change.Dim(); i++) {
      if (modified_objf_change(i) != objf_change_per_layer(i))
        scales(i) = sqrt(modified_objf_change(i) / objf_change_per_layer(i));
      else // need this if-statement mainly in case of zeros.
        scales(i) = 1.0;
    }

    Vector<BaseFloat> learning_rates(nu), modified_scales(nu);
    GetLearningRates(initial_nnet, scales, counter,
                     &learning_rates, &modified_scales);

    nnet_->SetLearningRates(learning_rates);

    Nnet diff_nnet(*nnet_);
    diff_nnet.AddNnet(-1.0, initial_nnet);
    // "diff_nnet" is the delta from start to now.
    Vector<BaseFloat> scale_delta(modified_scales);
    scale_delta.Add(-1.0);

    if (scale_delta.IsZero()) {
      KALDI_LOG << "Learning rates not being modified; objf change is "
                << tot_change << " vs. limit " << -max_degradation;
      return;
    }
    // Now "scale"
    const int num_operations = 4; // This number must be >= 1.  We break up the operation
                                  // of adding "diff_nnet" to *nnet into multiple operations,
                                  // so it would matter less if some of them get reverted
                                  // due to lack of locking.
    scale_delta.Scale(1.0 / num_operations);
    for (int32 k = 0; k < num_operations; k++)
      nnet_->AddNnet(scale_delta, diff_nnet);
    
    KALDI_LOG << "Objective-function change was " << objf_change_per_layer.Sum()
              << ", vs. allowed change " << -max_degradation;
    KALDI_LOG << "Objf changes per dim were " << objf_change_per_layer
              << ", limiting them to " << modified_objf_change
              << " by applying scales " << scales << " [modified to:]"
              << modified_scales << ", current learning rates are "
              << learning_rates;
    
  }
                           
  
  void CheckLearningRates(const Nnet &initial_nnet,
                          const std::vector<NnetTrainingExample> &egs,
                          int32 counter) {

    Nnet intermediate_nnet(initial_nnet);
    intermediate_nnet.Scale(0.5);
    intermediate_nnet.AddNnet(0.5, *nnet_);
    // now "intermediate_nnet" is halfway between "initial_nnet" and "nnet_", i.e.
    // halfway to the current parameter values.

    Nnet *nnet_gradient = new Nnet(*nnet_);
    const bool is_gradient = true;
    nnet_gradient->SetZero(is_gradient); // Set to zero and make it treated as a gradient.

    int32 batch_size = 512;  // This won't affect results, only efficiency.  This is probably
    // a reasonable value.

    ComputeNnetGradient(intermediate_nnet, egs, batch_size, nnet_gradient);

    Vector<BaseFloat> initial_dot_products(nnet_->NumUpdatableComponents()),
        current_dot_products(nnet_->NumUpdatableComponents());

    initial_nnet.ComponentDotProducts(*nnet_gradient, &initial_dot_products);
    nnet_->ComponentDotProducts(*nnet_gradient, &current_dot_products);
    
    Vector<BaseFloat> diff_dot_products(current_dot_products);
    diff_dot_products.AddVec(-1.0, initial_dot_products);
    diff_dot_products.Scale(1.0 / egs.size()); // To get the gradient per
                                               // testing example.
    delete nnet_gradient;
    nnet_gradient = NULL;
    
    KALDI_ASSERT(safe_config_.num_samples > 1);
    
    BaseFloat proportion_done =
        *counter_ / static_cast<BaseFloat>(safe_config_.num_samples);
    if (proportion_done > 1.1)
      KALDI_WARN << "Number of samples seen " << (*counter_)
                 << " exceeds --num-samples option " << safe_config_.num_samples
                 << " (indicates inaccurate --num-samples)";

    BaseFloat allowed_degradation =
        safe_config_.max_degradation * proportion_done;
    
    ModifyLearningRates(initial_nnet,
                        allowed_degradation,
                        diff_dot_products,
                        counter);
    
    KALDI_VLOG(2) << "Initial-dot-product is " << initial_dot_products
                  << ", current-dot-product is " << current_dot_products
                  << ", diff is " << diff_dot_products;
  }

  ~SafeBackpropClass() { }
 private:
  const SafeBackpropConfig &safe_config_;
  Nnet *nnet_;
  const int64 *counter_; // Keeps track of (approximately) how many examples
                         // we have already processed.
  ExamplesRepository *safe_backprop_repository_;
  ExamplesRepository *backprop_repository_;
  int32 backprop_minibatch_size_;

  BaseFloat momentum_minibatches_;
  Semaphore *minibatch_semaphore_;
  Mutex *done_mutex_;
  bool *done_;
};



BaseFloat DoBackpropParallelSafe(int32 minibatch_size,
                                 const SafeBackpropConfig &safe_config,
                                 SequentialNnetTrainingExampleReader *examples_reader,
                                 double *tot_weight,
                                 Nnet *nnet) {
  ExamplesRepository repository; // handles parallel programming issues
                                 // regarding the "examples" of data.
  ExamplesRepository safe_repository; // This a separate repository used by the
                                      // process that adjusts the learning
                                      // rates.
  
  double tot_log_prob = 0.0;
  int64 counter = 0;
  *tot_weight = 0;

  const bool store_separate_gradients = false;
  
  DoBackpropParallelClass bc(*nnet, &repository, tot_weight,
                             &tot_log_prob, nnet,
                             store_separate_gradients);

  SafeBackpropClass sc(safe_config, &safe_repository, &repository,
                       minibatch_size, nnet, &counter);
  
  {
    
    // g_num_threads - 1 threads are used for the actual learning.
    MultiThreader<DoBackpropParallelClass> m_backprop(g_num_threads - 1, bc);

    {
      // One thread is just for checking the learning rates.
      MultiThreader<SafeBackpropClass> m_safe_backprop(1, sc);

    
      std::vector<NnetTrainingExample> examples;
      std::vector<NnetTrainingExample> safe_examples;

      // We take every "modulus" frames and give it to the computation that
      // decides whether to limit the learning rates or not.
      const int32 min_modulus = 4; // this is arbitrary..
      int32 modulus = std::max(g_num_threads, min_modulus); 
      
      for (; !examples_reader->Done(); examples_reader->Next(), counter++) {
        if (counter % modulus == 0) safe_examples.push_back(examples_reader->Value());
        else  examples.push_back(examples_reader->Value());
      
        if (safe_examples.size() == safe_config.samples_per_test)
          safe_repository.AcceptExamples(&safe_examples);
        if (examples.size() == minibatch_size)
          repository.AcceptExamples(&examples);
      }
      safe_repository.ExamplesDone();
    
      // Get rid of any remaining examples in "safe_examples"
      while (!safe_examples.empty()) {
        examples.push_back(safe_examples.back());
        safe_examples.pop_back();
        if (examples.size() == minibatch_size)
          repository.AcceptExamples(&examples);
      }
      if (!examples.empty()) // partial minibatch.
        repository.AcceptExamples(&examples);
    }
    // 
    // the close-brace above will wait for the destructor of m_safe_backprop,
    // which will ensure that any remaining examples held there are written to
    // "repository", before calling repository.ExamplesDone().
    
    repository.ExamplesDone();
    // Here, the destructors of the MultiThreader objects re-join the threads.
  }
  KALDI_LOG << "Did backprop on " << *tot_weight << " examples, average log-prob "
            << "per frame is " << (tot_log_prob / *tot_weight);
  return tot_log_prob;
}


double DoBackpropSingleThreaded(const Nnet &nnet,
                                int32 minibatch_size,
                                const std::vector<NnetTrainingExample> &egs,
                                double *tot_weight,
                                Nnet *nnet_to_update) {
  double ans = 0.0;
  *tot_weight = TotalNnetTrainingWeight(egs);
  for (size_t i = 0; i < egs.size(); i += minibatch_size) {
    std::vector<NnetTrainingExample>::const_iterator end_iter =
      (i + minibatch_size > egs.size() ? egs.end() : 
       egs.begin() + i + minibatch_size);
    std::vector<NnetTrainingExample> this_egs(egs.begin() + i,
                                              end_iter);
    ans += DoBackprop(nnet, this_egs, nnet_to_update);
  }
  return ans;
}


double DoBackpropParallel(const Nnet &nnet,
                          int32 minibatch_size,
                          int32 num_threads,
                          const std::vector<NnetTrainingExample> &egs,
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
  KALDI_VLOG(2) << "Did backprop on " << *tot_weight << " examples, average log-prob "
                << "per frame is " << (tot_log_prob / *tot_weight);
  return tot_log_prob;
}

  
} // namespace nnet2
} // namespace kaldi
