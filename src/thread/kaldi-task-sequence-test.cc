// thread/kaldi-task-sequence-test.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
//                 Frantisek Skala

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


#include "base/kaldi-common.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi {

class MyTaskClass { // spins for a while, then outputs a pre-given integer.
 public:
  MyTaskClass(int32 i, std::vector<int32> *vec):
      done_(false), i_(i), vec_(vec) { }

  void operator() () {
    int32 spin = 1000000 * Rand() % 100;
    for (int32 i = 0; i < spin; i++);
    done_ = true;
  }
  ~MyTaskClass() {
    KALDI_ASSERT(done_);
    vec_->push_back(i_);
  }

 private:
  bool done_;
  int32 i_;
  std::vector<int32> *vec_;
};


void TestTaskSequencer() {
  TaskSequencerConfig config;
  config.num_threads = 1 + Rand() % 20;
  if (Rand() % 2 == 1 )
    config.num_threads_total = config.num_threads + Rand() % config.num_threads;

  int32 num_tasks = Rand() % 100;
  
  std::vector<int32> task_output;
  {
    TaskSequencer<MyTaskClass> sequencer(config);
    for (int32 i = 0; i < num_tasks; i++) {
      sequencer.Run(new MyTaskClass(i, &task_output));
    }
  } // and let "sequencer" be destroyed, which waits for the last threads.
  KALDI_ASSERT(task_output.size() == static_cast<size_t>(num_tasks));
  for (int32 i = 0; i < num_tasks; i++)
    KALDI_ASSERT(task_output[i] == i);
}

}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 1000; i++)
    TestTaskSequencer();
}

