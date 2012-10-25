// util/kaldi-thread.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
//                 Frantisek Skala

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

#ifndef KALDI_THREAD_KALDI_THREAD_H_
#define KALDI_THREAD_KALDI_THREAD_H_ 1

#include <pthread.h>
#include "thread/kaldi-barrier.h"
// This header provides a convenient mechanism for parallelization.  The idea is
// that you have some range of integers, e.g. A ... B-1 (with B > A), and some
// function call that takes a range of integers, and you partition these up into
// a number of blocks.

// TODO: if needed, provide a workaround for Windows and other
// non-POSIX-compliant systems, possibly one that does not actually do
// multi-threading.


// Description of MultiThreadPool and it's usage:
//
// Usage of the RunMultiThreadedPersistent is the same as the usage of
// RunMultiThreaded, except that the object provided ust inherit MultiThreadable
// and it's run method isn't called, but operator() is called directly instead.
// Member variables num_threads_ and thread_id_ must NOT be redefined in the
// classes used, as they are called when using MultiThreadable*
//
// MultiThreadPool is a singleton class, it's instance is obtained using
// MultiThreadPool::Instantiate(). First instantiation initializes the thread
// pool using g_num_threads threads, each of those threads runs infinite loop in
// ThreadWorker::run(). When RunMultiThreadedPersistent(c) is called, each
// ThreadWorker is given a pointer to a copy of c and calls c() in it's thread.
// After doing this, it Waits on barrier to sync with all the threads and the
// main one, then Waits again until it receives another job.

namespace kaldi {

extern int32 g_num_threads;  // Maximum number of threads (for programs that
// use threads, which is not many of them, e.g. the SGMM update program does.
// This is 8 by default.  You can change this on the command line, where
// used, with --num-threads.  Programs that think they will use threads
// should register it with their ParseOptions, as something like:
// po.Register("num-threads", &g_num_threads, "Number of threads to use.");

class MultiThreadable {
  // To create function that does part of the job, create class that inherits
  // this one, reimplements operator() and does part of the job based on
  //  thread_id_ and num_threads_
  // Note: example implementations are in thread/kaldi-thread-test.cc
 public:
  virtual void operator() () = 0;
  // Does the main function of the class
  //  Subclasses have to redefine this
  virtual ~MultiThreadable();
  // Optional destructor.  Note: the destructor
  // the object passed by the user will also be called, so
  // watch out.

  static void *run(void *m_in) {
    MultiThreadable *m = static_cast<MultiThreadable*>(m_in);
    (*m)();  // call operator () on it.  This is a virtual
    // function so the one in the child class will be called.
    return NULL;
  }
  
 public:
  // Do not redeclare thread_id_ and num_threads_ in derived classes.
  int32 thread_id_;  // 0 <= thread_id_ < num_threads_
  int32 num_threads_;

 private:
  // Have additional member variables as needed.
};


class ExampleClass: public MultiThreadable {
 public:
  ExampleClass(const ExampleClass &other) {
    // .. optional initalizer.  Run sequentially; each of the parallel
    // ExampleClass members that we'll run in parallel will in turn
    // be initialized from the object passed by user.
  }
  void operator() () {
    // Does the main function of the class.  This
    // function will typically want to look at the values of the 
    // member variables thread_id_ and num_threads_, inherited
    // from MultiThreadable.
  }
  ~ExampleClass() {
    // Optional destructor.  Sometimes useful things happen here,
    // for example summing up of certain quantities.  See code
    // that uses RunMultiThreaded for examples.
  }
 private:
  // Have additional member variables as needed.
};

/// RunMultiThreaded takes a class C similar to ExampleClass above, and runs the
/// code inside it in parallel; it waits till all threads are done and then
/// returns.  The number of threads used is g_num_threads.
template<class C> void RunMultiThreaded(const C &c_in) {
  KALDI_ASSERT(g_num_threads > 0);
  if (g_num_threads == 1) {  // Just run one copy.
    C c(c_in);  // create a copy of the object, just for consistency
    c.thread_id_ = 0;
    c.num_threads_ = 1;
    // with what happens in the multi-threaded case.
    C::run(&c);  // Note: this is the same as calling c(), but
    // we do it like this in case the user (ill-advisedly) put any
    // other statements in the static "run" function.
  } else {
    pthread_t *threads = new pthread_t[g_num_threads];
    std::vector<C> cvec(g_num_threads, c_in);  // all initialized with same
    // object.
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);
    for (int32 thread = 0; thread < g_num_threads; thread++) {
      cvec[thread].thread_id_ = thread;
      cvec[thread].num_threads_ = g_num_threads;
      int32 ret;
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
    delete [] threads;
  }
}

} // namespace kaldi
#endif  // KALDI_THREAD_KALDI_THREAD_H_
