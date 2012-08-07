// util/kaldi-barrier.h

// Copyright 2012  Karel Vesely (Brno University of Technology)

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


#ifndef KALDI_UTIL_KALDI_BARRIER_H_
#define KALDI_UTIL_KALDI_BARRIER_H_ 1


#include <pthread.h>


namespace kaldi {

/**
 * The Barrier class
 * A barrier causes a group of threads to wait until 
 * all the threads reach the "barrier".
 */
class Barrier {
 public:
  Barrier(int threshold=0);
  ~Barrier();

  void SetThreshold(int thr); ///< number of threads to wait for
  int Wait(); ///< last thread returns -1, the others 0

 private:
  pthread_mutex_t     mutex_;     ///< Mutex which control access to barrier 
  pthread_cond_t      cv_;        ///< Conditional variable to make barrier wait

  int                 threshold_; ///< size of thread-group
  int                 counter_;   ///< number of threads we wait for
  int                 cycle_;     ///< cycle flag to keep synchronized

};

} // namespace kaldi

#endif // KALDI_UTIL_KALDI_BARRIER_H_

