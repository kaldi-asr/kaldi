// util/kaldi-thread-test.cc

// Copyright 2012  Daniel Povey

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


#include "base/kaldi-common.h"
#include "util/kaldi-thread.h"



namespace kaldi {

class MyThreadClass { // Sums up integers from 0 to max_to_count-1.
 public:
  MyThreadClass(int32 max_to_count, int32 *i): max_to_count_(max_to_count),
                                               iptr_(i),
                                               private_counter_(0) { }
  // Use default copy constructor and assignment operators.
  void operator () () {
    int block_size = (max_to_count_+ (num_threads_-1) ) / num_threads_;
    int start = block_size * thread_id_,
        end = std::min(max_to_count_, start + block_size);
    for (int j = start; j < end; j++)
      private_counter_ += j*j;
  }
  ~MyThreadClass() {
    *iptr_ += private_counter_;
  }
  
  static void *run(void *c_in) {
    MyThreadClass *c = static_cast<MyThreadClass*>(c_in);
    (*c)(); // call operator () on it.
    return NULL;
  }  
  
 public:
  int thread_id_; // 0 <= thread_number < num_threads
  int num_threads_;
  
 private:
  MyThreadClass() { };  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
};


void TestThreads() {
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot = (10000*(10000-1))/2);
  }
  g_num_threads = 1;
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot = (10000*(10000-1))/2);
  }
  g_num_threads = 8;
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  TestThreads();
}

