// thread/kaldi-thread.h

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

#ifndef KALDI_THREAD_KALDI_THREAD_H_
#define KALDI_THREAD_KALDI_THREAD_H_ 1

#if defined(_MSC_VER)
# define KALDI_PTHREAD_PTR(thread) (thread.p)
#else
# define KALDI_PTHREAD_PTR(thread) (thread)
#endif

#include <pthread.h>
#include "thread/kaldi-barrier.h"
// This header provides a convenient mechanism for parallelization.  The idea is
// that you have some range of integers, e.g. A ... B-1 (with B > A), and some
// function call that takes a range of integers, and you partition these up into
// a number of blocks.
// Also see kaldi-task-sequence.h which is suitable for parallelizing the processing
// of tasks coming in sequentially from somewhere.

// TODO: if needed, provide a workaround for Windows and other
// non-POSIX-compliant systems, possibly one that does not actually do
// multi-threading.


// Description of MultiThreadPool and its usage:
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
  ExampleClass(int32 *foo); // Typically there will be an initializer that
  // takes arguments.

  ExampleClass(const ExampleClass &other); // A copy constructor is also needed;
  // some example classes use the default version of this.

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


template<class C>
class MultiThreader {
 public:
  MultiThreader(int32 num_threads,
                const C &c_in):
    threads_(new pthread_t[std::max<int32>(1, num_threads)]),
    cvec_(std::max<int32>(1, num_threads), c_in) {
    if (num_threads == 0) {
      // This is a special case with num_threads == 0, which behaves like with
      // num_threads == 1 but without creating extra threads.  This can be
      // useful in GPU computations where threads cannot be used.
      KALDI_PTHREAD_PTR(threads_[0]) = 0;
      cvec_[0].thread_id_ = 0;
      cvec_[0].num_threads_ = 1;
      (cvec_[0])();
    } else {
      pthread_attr_t pthread_attr;
      pthread_attr_init(&pthread_attr);
      for (int32 thread = 0; thread < num_threads; thread++) {
        cvec_[thread].thread_id_ = thread;
        cvec_[thread].num_threads_ = num_threads;
        int32 ret;
        if ((ret=pthread_create(&(threads_[thread]),
                                &pthread_attr, C::run, &(cvec_[thread])))) {
          const char *c = strerror(ret);
          if (c == NULL) { c = "[NULL]"; }
          KALDI_ERR << "Error creating thread, errno was: " << c;
        }
      }
    }
  }
  ~MultiThreader() {
    for (size_t thread = 0; thread < cvec_.size(); thread++)
      if (KALDI_PTHREAD_PTR(threads_[thread]) != 0)
        if (pthread_join(threads_[thread], NULL))
          KALDI_ERR << "Error rejoining thread.";
    delete [] threads_;
  }
 private:
  pthread_t *threads_;
  std::vector<C> cvec_;
};

/// Here, class C should inherit from MultiThreadable.  Note: if you want to
/// control the number of threads yourself, or need to do something in the main
/// thread of the program while the objects exist, just initialize the
/// MultiThreader<C> object yourself.
template<class C> void RunMultiThreaded(const C &c_in) {
  MultiThreader<C> m(g_num_threads, c_in);
}



} // namespace kaldi
#endif  // KALDI_THREAD_KALDI_THREAD_H_
