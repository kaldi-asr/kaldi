// thread/kaldi-barrier.cc

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


#include <pthread.h>
#include "base/kaldi-error.h"
#include "thread/kaldi-barrier.h"


namespace kaldi {



Barrier::Barrier(int32 threshold)
 : threshold_(threshold), counter_(threshold), cycle_(0) {

  if(0 != pthread_mutex_init(&mutex_, NULL))
    KALDI_ERR << "Cannot initialize pthread mutex";
  
  if(0 != pthread_cond_init(&cv_, NULL)) {
    pthread_mutex_destroy(&mutex_);
    KALDI_ERR << "Cannot initialize pthread condv";
  }
}



Barrier::~Barrier() {
  if(0 != pthread_mutex_lock(&mutex_))
    KALDI_ERR << "Cannot lock pthread mutex";

  if(counter_ != threshold_) {
    pthread_mutex_unlock (&mutex_);
    KALDI_ERR << "Cannot destroy barrier with waiting thread(s)";
  }

  if(0 != pthread_mutex_unlock(&mutex_))
    KALDI_ERR << "Cannot unlock pthread mutex";

  if(0 != pthread_mutex_destroy(&mutex_))
    KALDI_ERR << "Cannot destroy pthread mutex";

  if(0 != pthread_cond_destroy(&cv_)) 
    KALDI_ERR << "Cannot destroy pthread condv";
}



void Barrier::SetThreshold(int32 thr) {
  if(counter_ != threshold_) {
    KALDI_ERR << "Cannot set threshold, while some thread(s) are waiting";
  }
  threshold_ = thr; counter_ = thr;
}



/**
 * Wait for all the threads to reach a barrier. 
 * A broadcast wakes all the waiting threads when the counter_ reaches 0.
 * The last incoming thread returns -1, the others 0.
 */
int32 Barrier::Wait() {
  if(threshold_ == 0)
    KALDI_ERR << "Cannot wait when ``threshold'' value was not set";

  if(0 != pthread_mutex_lock(&mutex_)) 
    KALDI_ERR << "Cannot lock pthread mutex";

  int32 cycle = cycle_;   // memorize which cycle we're in 

  int32 ret;
  if(--counter_ == 0) { ///<< THIS IS LAST THREAD
    cycle_ = !cycle_;
    counter_ = threshold_;
    // WAKE-UP!!!!!!!!!!!!!!!!!!
    if(0 != pthread_cond_broadcast(&cv_)) { 
      KALDI_ERR << "Error on pthred_cond_broadcast";
    }
    /*
     * The last incoming thread will return -1, the others 0
     */
    ret = -1;

  } else { ///<< NOT A LAST THTREAD
    /*
     * Wait with thread cancellation disabled, barrier waiting
     * should not be a cancellation point.
     */
    int32 cancel, tmp;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel);

    /*
     * Wait until the barrier's cycle_ changes, like this we know
     * that the last thread arrived and a brodcast was sent.
     */
    while (cycle == cycle_) {
      // GO-SLEEP, zzzzzzzzzzz...
      if(0 != pthread_cond_wait(&cv_, &mutex_)) {
        KALDI_ERR << "Error on pthread_cond_wait";
      }
    }
 
    /*
     * Restore the threads' cancel state
     */
    pthread_setcancelstate(cancel, &tmp);
 
    /*
     * Indicate that it is not the last thread
     */
    ret = 0; 
  }

  if(0 != pthread_mutex_unlock (&mutex_)) {
    KALDI_ERR << "Error unlcoking pthread mutex";
  }

  return ret; 
}


} // namespace kaldi
