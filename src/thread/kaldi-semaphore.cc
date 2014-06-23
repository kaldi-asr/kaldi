// thread/kaldi-semaphore.cc

// Copyright 2012  Karel Vesely (Brno University of Technology)

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



#include "base/kaldi-error.h"
#include "thread/kaldi-semaphore.h"

namespace kaldi {

  
Semaphore::Semaphore(int32 initValue) {
  counter_ = initValue;
  if (pthread_mutex_init(&mutex_, NULL) != 0) {
    KALDI_ERR << "Cannot initialize pthread mutex";
  }
  if (pthread_cond_init(&cond_, NULL) != 0) {
    KALDI_ERR << "Cannot initialize pthread conditional variable";
  }
}



Semaphore::~Semaphore() {
  if (pthread_mutex_destroy(&mutex_) != 0) {
    KALDI_ERR << "Cannot destroy pthread mutex";
  }
  if (pthread_cond_destroy(&cond_) != 0) {
    KALDI_ERR << "Cannot destroy pthread conditional variable";
  }
}



bool Semaphore::TryWait() {
  int32 ret = 0;
  bool try_wait_succeeded = false;
  ret |= pthread_mutex_lock(&mutex_);
  if (counter_ > 0) {
    counter_--;
    try_wait_succeeded = true;
  }
  ret |= pthread_mutex_unlock(&mutex_);
  if (ret != 0) {
    KALDI_ERR << "Error in pthreads";
  }
  return try_wait_succeeded;
}



void Semaphore::Wait() {
  int32 ret = 0;
  ret |= pthread_mutex_lock(&mutex_);
  while (counter_ <= 0) {
    ret |= pthread_cond_wait(&cond_, &mutex_);
  }
  counter_--;
  ret |= pthread_mutex_unlock(&mutex_);
  if (ret != 0) {
    KALDI_ERR << "Error in pthreads";
  }
}



void Semaphore::Signal() {
  int32 ret = 0;
  ret |= pthread_mutex_lock(&mutex_);
  counter_++;
  ret |= pthread_cond_signal(&cond_);
  ret |= pthread_mutex_unlock(&mutex_);
  if (ret != 0) {
    KALDI_ERR << "Error in pthreads";
  }
}


} // namespace kaldi
