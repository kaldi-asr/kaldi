// thread/kaldi-barrier.h

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


#ifndef KALDI_THREAD_KALDI_BARRIER_H_
#define KALDI_THREAD_KALDI_BARRIER_H_ 1


#include <pthread.h>


namespace kaldi {

/**
 * The Barrier class
 * A barrier causes a group of threads to wait until 
 * all the threads reach the "barrier".
 */
class Barrier {
 public:
  Barrier(int32 threshold=0);
  ~Barrier();

  void SetThreshold(int32 thr); ///< number of threads to wait for
  int32 Wait(); ///< last thread returns -1, the others 0

 private:
  pthread_mutex_t     mutex_;     ///< Mutex which control access to barrier 
  pthread_cond_t      cv_;        ///< Conditional variable to make barrier wait

  int32                 threshold_; ///< size of thread-group
  int32                 counter_;   ///< number of threads we wait for
  int32                 cycle_;     ///< cycle flag to keep synchronized

};

} // namespace kaldi

#endif // KALDI_THREAD_KALDI_BARRIER_H_

