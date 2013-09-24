// thread/kaldi-mutex.cc

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
#include <cerrno>

#include "base/kaldi-error.h"
#include "thread/kaldi-mutex.h"

namespace kaldi {
  

Mutex::Mutex() {
  if(0 != pthread_mutex_init(&mutex_,NULL)) 
    KALDI_ERR << "Cannot initialize pthread mutex";
}


Mutex::~Mutex() {
  if(0 != pthread_mutex_destroy(&mutex_)) 
    KALDI_ERR << "Cannot destroy pthread mutex";
}


void Mutex::Lock() {
  if(0 != pthread_mutex_lock(&mutex_))
    KALDI_ERR << "Error on locking pthread mutex";
}

 
bool Mutex::TryLock() {
  int32 ret = pthread_mutex_trylock(&mutex_);
  bool lock_succeeded = false;
  switch (ret) {
    case 0: lock_succeeded = true;
    case EBUSY: lock_succeeded = false;
    default: KALDI_ERR << "Error on try-locking pthred mutex";
  }
  return lock_succeeded;
}


void Mutex::Unlock() {
  if(0 != pthread_mutex_unlock(&mutex_))
    KALDI_ERR << "Error on unlocking pthread mutex";
}


  
} // namespace kaldi

