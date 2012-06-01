// util/kaldi-thread.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_BASE_KALDI_THREAD_H_
#define KALDI_BASE_KALDI_THREAD_H_ 1

#include <pthread.h>
// This header provides a convenient mechanism for parallelization.  The idea is
// that you have some range of integers, e.g. A ... B-1 (with B > A), and some
// function call that takes a range of integers, and you partition these up into
// a number of blocks.

// TODO: if needed, provide a workaround for Windows and other non-POSIX-compliant
// systems, possibly one that does not actually do multi-threading.


namespace kaldi {

extern int32 g_num_threads; // Maximum number of threads (for programs that
// use threads, which is not many of them, e.g. the SGMM update program does.
// This is 8 by default.  You can change this on the command line, where
// used, with --num-threads.  Programs that think they will use threads
// should register it with their ParseOptions, as something like:
// po.Register("num-threads", &g_num_threads, "Number of threads to use.");

class ExampleClass {
 public:
  ExampleClass (const ExampleClass &other) {
    // .. optional initalizer.  Run sequentially;
    // initialized from object passed by user.
  }
  void operator() (){
    // Does the main function of the class
  }
  ~ExampleClass() {
    // Optional destructure.  Note: the destructor
    // the object passed by the user will also be called, so
    // watch out.
  }

  // This function should be provided. Give it this exact implementation, with
  // the class name replaced with your own class's name.
  static void *run(void *c_in) {
    ExampleClass *c = static_cast<ExampleClass*>(c_in);
    (*c)(); // call operator () on it.
    return NULL;
  }  
 public:
  int thread_id_; // 0 <= thread_number < num_threads
  int num_threads_;
  
 private:
  // Have additional member variables as needed.
};

template<class C> void RunMultiThreaded(const C &c_in) {
  KALDI_ASSERT(g_num_threads > 0);
  if (g_num_threads == 1) { // Just run one copy.
    C c(c_in); // create a copy of the object, just for consistency
    c.thread_id_ = 0;
    c.num_threads_ = 1;
    // with what happens in the multi-threaded case.
    C::run(&c); // Note: this is the same as calling c(), but
    // we do it like this in case the user (ill-advisedly) put any
    // other statements in the static "run" function.
  } else {
    pthread_t *threads = new pthread_t[g_num_threads];
    std::vector<C> cvec(g_num_threads, c_in); // all initialized with same object.
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);
    for (int32 thread = 0; thread < g_num_threads; thread++) {
      cvec[thread].thread_id_ = thread;
      cvec[thread].num_threads_ = g_num_threads;
      int ret;
      if ((ret=pthread_create(&(threads[thread]),
                              &pthread_attr, C::run, &(cvec[thread])))) {
        const char *c = strerror(ret);
        if (c == NULL) { c = "[NULL]"; }
        KALDI_ERR << "Error creating thread, errno was: " << c;
      }
    }
    for (int32 thread = 0; thread < g_num_threads; thread++)
      if (pthread_join(threads[thread], NULL))
        KALDI_ERR << "Error rejoining thread.";
    delete threads;
  }
}

}

#endif  // KALDI_BASE_KALDI_THREAD_H_
