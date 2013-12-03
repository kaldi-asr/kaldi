// thread/kaldi-task-sequence.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_THREAD_KALDI_TASK_SEQUENCE_H_
#define KALDI_THREAD_KALDI_TASK_SEQUENCE_H_ 1

#include <pthread.h>
#include "thread/kaldi-thread.h"
#include "itf/options-itf.h"
#include "thread/kaldi-semaphore.h"


namespace kaldi {

/**
   In kaldi-thread.h, the class MultiThreader, and the function
   RunMultiThreaded, provided a mechanism to run a specified number of jobs
   simultaneously, in parellel, and wait for them all to finish.  This file
   addresses a different problem, typically encountered in Kaldi command-line
   programs that process a sequence of items.  The problem is where
   items to be processed are coming in, all of different sizes (e.g. utterances
   with different numbers of frames) and we would like them to be run in
   parallel somehow, using multiple threads; but they must still make good use
   of the number of threads available; and they must be output in the same order
   as they came in.

   Here, we will still accept objects of some class C with an operator () that
   takes no arguments.  C may also have a constructor and a destructor that do
   something (typically the constructor just sets variables, and the destructor
   does some kind of output).  We have a templated class TaskSequencer<C> which
   is responsible for running the jobs in parallel.  It has a function Run()
   that will accept a new object of class C; this will block until a thread is
   free, at which time it will spawn a thread that starts running the operator
   () of the class.  When classes are finished running, the objects will be
   deleted.  Class TaskSequencer guarantees that the destructors will be called
   sequentially (not in parallel) and in the same order the objects were given
   to the Run() function, so that it is safe for the destructor to have side
   effects such as outputting data.

   Note: the destructor of TaskSequencer will wait for any remaining jobs that
   are still running and will call the destructors.   
 */

struct TaskSequencerConfig {
  int32 num_threads;
  int32 num_threads_total;
  TaskSequencerConfig(): num_threads(1), num_threads_total(0)  { }
  void Register(OptionsItf *po) {
    po->Register("num-threads", &num_threads, "Number of actively processing "
                 "threads to run in parallel");
    po->Register("num-threads-total", &num_threads_total, "Total number of "
                 "threads, including those that are waiting on other threads "
                 "to produce their output.  Controls memory use.  If <= 0, "
                 "defaults to --num-threads plus 20.  Otherwise, must "
                 "be >= num-threads.");
  }
};

// C should have an operator () taking no arguments, that does some kind
// of computation, and a destructor that produces some kind of output (the
// destructors will be run sequentially in the same order Run as called.
template<class C>
class TaskSequencer {
 public:
  TaskSequencer(const TaskSequencerConfig &config):
      threads_avail_(config.num_threads),
      tot_threads_avail_(config.num_threads_total > 0 ? config.num_threads_total :
                         config.num_threads + 20),
      thread_list_(NULL) {
    KALDI_ASSERT((config.num_threads_total <= 0 ||
                  config.num_threads_total >= config.num_threads) &&
                 "num-threads-total, if specified, must be >= num-threads");
  }

  /// This function takes ownership of the pointer "c", and will delete it
  /// in the same sequence as Run was called on the jobs.
  void Run(C *c) {
    threads_avail_.Wait(); // wait till we have a thread for computation free.
    tot_threads_avail_.Wait(); // this ensures we don't have too many threads
    // waiting on I/O, and consume too much memory.
    
    // put the new RunTaskArgsList object at head of the singly
    // linked list thread_list_.
    thread_list_ = new RunTaskArgsList(this, c, thread_list_);
    int32 ret;
    if ((ret=pthread_create(&(thread_list_->thread),
                            NULL, // default attributes
                            TaskSequencer<C>::RunTask,
                            static_cast<void*>(thread_list_)))) {
      const char *c = strerror(ret);
      KALDI_ERR << "Error creating thread, errno was: " << (c ? c : "[NULL]");
    }
  }

  void Wait() { // You call this at the end if it's more convenient
    // than waiting for the destructor.  It waits for all tasks to finish.
    if (thread_list_ != NULL) {
      int ret = pthread_join(thread_list_->thread, NULL);
      if (ret != 0) {
        const char *c = strerror(ret);
        KALDI_ERR << "Error joining thread, errno was: " << (c ? c : "[NULL]");
      }
      KALDI_ASSERT(thread_list_->tail == NULL); // thread would not
      // have exited without setting tail to NULL.
      delete thread_list_;
      thread_list_ = NULL;
    }
  }
  
  /// The destructor waits for the last thread to exit.
  ~TaskSequencer() {
    Wait();      
  }
 private:
  struct RunTaskArgsList {
    TaskSequencer *me; // Think of this as a "this" pointer.
    C *c; // Clist element of the task we're expected
    pthread_t thread;
    RunTaskArgsList *tail;
    RunTaskArgsList(TaskSequencer *me, C *c, RunTaskArgsList *tail):
        me(me), c(c), tail(tail) {}
  };
  // This static function gets run in the threads that we create.
  static void* RunTask(void *input) {
    RunTaskArgsList *args = static_cast<RunTaskArgsList*>(input);
    
    // (1) run the job.
    (*(args->c))(); // call operator () on args->c, which does the computation.
    args->me->threads_avail_.Signal(); // Signal that the compute-intensive
    // part of the thread is done (we want to run no more than
    // config_.num_threads of these.)
    
    // (2) we want to destroy the object "c" now, by deleting it.  But for
    //     correct sequencing (this is the whole point of this class, it
    //     is intended to ensure the output of the program is in correct order),
    //     we first wait till the previous thread, whose details will be in "tail",
    //     is finished.
    if (args->tail != NULL) {
      int ret = pthread_join(args->tail->thread, NULL);
      if (ret != 0) {
        const char *c = strerror(ret);
        KALDI_ERR << "Error joining thread, errno was: " << (c ? c : "[NULL]");
      }
    }

    delete args->c; // delete the object "c".  This may cause some output,
    // e.g. to a stream.  We don't need to worry about concurrent access to
    // the output stream, because each thread waits for the previous thread
    // to be done, before doing this.  So there is no risk of concurrent
    // access.
    args->c = NULL;
    
    if (args->tail != NULL) {
      KALDI_ASSERT(args->tail->tail == NULL); // Because we already
      // did pthread_join on args->tail->thread, which means that
      // thread was done, and before it exited, it would have
      // deleted and set to NULL its tail (which is the next line of code).
      delete args->tail;
      args->tail = NULL;
    }
    // At this point we are exiting from the thread.  Signal the
    // "tot_threads_avail_" semaphore which is used to limit the total number of threads that are alive, including
    // not onlhy those that are in active computation in c->operator (), but those
    // that are waiting on I/O or other threads.
    args->me->tot_threads_avail_.Signal();
    // .. and exit the thread, by returning. 
    return NULL;
  }

  Semaphore threads_avail_; // Initialized to the number of threads we are
  // supposed to run with; the function Run() waits on this.

  Semaphore tot_threads_avail_; // We use this semaphore to ensure we don't
  // consume too much memory...
  RunTaskArgsList *thread_list_; 
  
};

} // namespace kaldi

#endif  // KALDI_THREAD_KALDI_TASK_SEQUENCE_H_
