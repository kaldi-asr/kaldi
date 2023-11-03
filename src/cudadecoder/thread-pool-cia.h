// cudadecoder/thread-pool-cia.h
//
// Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
// Daniel Galvez
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This code was modified from Chapter 10 of C++ Concurrency in
// Action, which offers its code under the Boost License.

#pragma once

#include <cassert>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <queue>
#include <vector>

#ifdef __linux__
#include <nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif // __linux__


namespace kaldi {

class join_threads {
  std::vector<std::thread>& threads;
public:
  explicit join_threads(std::vector<std::thread>& threads_): threads(threads_) 
  {}
  ~join_threads() {
    for (unsigned int i = 0; i < threads.size(); ++i) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }
  }
};

template<typename T>
class threadsafe_queue {
private:
  mutable std::mutex mut;
  std::queue<T> data_queue;
  std::condition_variable data_cond;
  std::atomic<bool> done;

public:
  threadsafe_queue(): done(false) {}
  threadsafe_queue& operator=(const threadsafe_queue&) = delete;

  void mark_done() {
    std::lock_guard<std::mutex> lk(mut);
    done = true;
    data_cond.notify_all();
  }

  ~threadsafe_queue() {
    if (!done) {
      assert(false && "Must set to done to true before destroying threadsafe_queue.");
    }
  }

  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_move_assignable<T>::value, void>::type
  push(T new_value) {
    std::lock_guard<std::mutex> lk(mut);
    // There appears to be no reason not to use std::move here...
    data_queue.push(std::move(new_value));
    data_cond.notify_one();
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_copy_assignable<T>::value && !std::is_move_assignable<T>::value, void>::type
  push(T new_value) {
    std::lock_guard<std::mutex> lk(mut);
    // There appears to be no reason not to use std::move here...
    data_queue.push(new_value);
    data_cond.notify_one();
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_move_assignable<T>::value, bool>::type
  wait_and_pop(T& value)
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this]{return !data_queue.empty() || done;});
    if (!data_queue.empty()) {
      value = std::move(data_queue.front());
      data_queue.pop();
      return true;
    } else {
      return false;
    }
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_copy_assignable<T>::value && !std::is_move_assignable<T>::value, bool>::type
  wait_and_pop(T& value)
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this]{return !data_queue.empty() || done;});
    if (!data_queue.empty()) {
      value = data_queue.front();
      data_queue.pop();
      return true;
    } else {
      return false;
    }
  }
  // TODO: return null pointer if done. TODO: Add move assign overload.
  std::unique_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this]{return !data_queue.empty();});
    std::unique_ptr<T> res(std::make_unique<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_move_assignable<T>::value, bool>::type
  try_pop(T& value) {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty()) {
      return false;
    }
    value = std::move(data_queue.front());
    data_queue.pop();
    return true;
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_copy_assignable<T>::value && !std::is_move_assignable<T>::value, bool>::type
  try_pop(T& value) {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty()) {
      return false;
    }
    value = data_queue.front();
    data_queue.pop();
    return true;
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_move_assignable<T>::value, std::unique_ptr<T>>::type
  try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return std::unique_ptr<T>();
    }
    std::unique_ptr<T> res(std::make_unique<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }
  template<class U = T>
  typename std::enable_if<std::is_same<T,U>::value && std::is_copy_assignable<T>::value && !std::is_move_assignable<T>::value, std::unique_ptr<T>>::type
  try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return std::unique_ptr<T>();
    }
    std::unique_ptr<T> res(std::make_unique<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }
  bool empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }
};

class thread_pool {
  std::atomic_bool done;
  threadsafe_queue<std::function<void()>> work_queue;
  std::vector<std::thread> threads;
  join_threads joiner;
  // class PassKey {
  //   friend class thread_pool;
  //   PassKey() = default;
  //   ~PassKey() = default;
  // };
public:
  void worker_thread(/*PassKey*/) {
    while (!done) {
      std::function<void()> task;
      // wait_and_pop seems more efficient than try_pop...
      if (work_queue.try_pop(task)) {
        task();
      } else {
        std::this_thread::yield();
      }
    }
  }
  thread_pool(unsigned int const num_threads): done(false), joiner(threads) {
    try {
      for (unsigned int i = 0; i < num_threads;++i) {
        threads.push_back(std::thread(&thread_pool::worker_thread, this/*, PassKey()*/));
      }
    } catch(...) {
      done = true;
      throw;
    }
  }

  ~thread_pool() {
    done = true;
  }

  template<typename FunctionType>
  void submit(FunctionType f) {
    work_queue.push(std::function<void()>(f));
  }
};


// 9.2

class function_wrapper {
  struct impl_base {
    virtual void call()=0;
    virtual ~impl_base() {}
  };
  std::unique_ptr<impl_base> impl;
  template<typename F>
  struct impl_type: impl_base
  {
    F f;
    impl_type(F&& f_): f(std::move(f_)) {}
    void call() { f(); }
  };

public:
  template<typename F>
  function_wrapper(F&& f): impl(new impl_type<F>(std::move(f))) {}
  void operator()() {impl->call(); }
  function_wrapper() = default;
  function_wrapper(function_wrapper&& other): impl(std::move(other.impl)) {}
  function_wrapper& operator=(function_wrapper&& other)
  {
    impl = std::move(other.impl);
    return *this;
  }
  function_wrapper(const function_wrapper&) = delete;
  function_wrapper(function_wrapper&) = delete;
  function_wrapper& operator=(const function_wrapper&) = delete;
};


class futures_thread_pool {
  std::atomic_bool done;
  threadsafe_queue<function_wrapper> work_queue;
  std::vector<std::thread> threads;
  join_threads joiner;

public:
  void worker_thread() {
    #ifdef __linux__
    nvtxNameOsThread(syscall(SYS_gettid), "threadpool");
    pthread_setname_np(pthread_self(), "threadpool");
    #endif
    while (!done) {
      function_wrapper task;
      bool success = work_queue.wait_and_pop(task);
      if (success) {
        task();
      }
      // if (work_queue.try_pop(task)) {
      //   task();
      // } else {
      //   std::this_thread::yield();
      // }
    }
  }
  futures_thread_pool(const unsigned int num_threads): done(false), joiner(threads) {
    try {
      for (unsigned int i = 0; i < num_threads;++i) {
        threads.push_back(std::thread(&futures_thread_pool::worker_thread, this));
      }
    } catch(...) {
      done = true;
      throw;
    }
  }

  ~futures_thread_pool() {
    work_queue.mark_done();
    done = true;
  }

  // can we include Args... args as well here? Don't think so...
  template<typename FunctionType>
  std::future<typename std::result_of<FunctionType()>::type>
  submit(FunctionType f) {
    typedef typename std::result_of<FunctionType()>::type result_type;
    std::packaged_task<result_type()> task(std::move(f));
    std::future<result_type> res(task.get_future());
    work_queue.push(std::move(task));
    return res;
  }

  size_t num_workers() const {
    return threads.size();
  }
};

class thread_local_queue_thread_pool {
  std::atomic_bool done;
  std::vector<std::thread> threads;
  join_threads joiner;
  threadsafe_queue<function_wrapper> pool_work_queue;
  typedef std::queue<function_wrapper> local_queue_type;
  // why unique_ptr here?
  static thread_local std::unique_ptr<local_queue_type> local_work_queue;
  void run_pending_task() {
    function_wrapper task;
    if (local_work_queue && !local_work_queue->empty()) {
      task = std::move(local_work_queue->front());
      local_work_queue->pop();
      task();
    } else if (pool_work_queue.try_pop(task)) {
      task();
    } else {
      std::this_thread::yield();
    }
  }

public:
  void worker_thread() {
    local_work_queue.reset(new local_queue_type);

    // spining here, unlike previous implementation...
    while (!done) {
      run_pending_task();
    }
  }

  thread_local_queue_thread_pool(unsigned int const num_threads): done(false), joiner(threads) {
    try {
      for (unsigned int i = 0; i < num_threads;++i) {
        threads.push_back(std::thread(&thread_local_queue_thread_pool::worker_thread, this));
      }
    } catch(...) {
      done = true;
      throw;
    }
  }

  ~thread_local_queue_thread_pool() {
    done = true;
  }

  template<typename FunctionType>
  std::future<typename std::result_of<FunctionType()>::type>
  submit(FunctionType f) {
    typedef typename std::result_of<FunctionType()>::type result_type;
    std::packaged_task<result_type()> task(f);
    std::future<result_type> res(task.get_future());
    if(local_work_queue) {
      local_work_queue->push(std::move(task));
    } else {
      pool_work_queue.push(std::move(task));
    }
    return res;
  }
};

class work_stealing_queue {
private:
  typedef function_wrapper data_type;
  std::deque<data_type> the_queue;
  mutable std::mutex the_mutex;

public:
  work_stealing_queue() {}
  work_stealing_queue(const work_stealing_queue& other) = delete;
  work_stealing_queue& operator=(const work_stealing_queue& other) = delete;
  void push(data_type data)
  {
    std::lock_guard<std::mutex> lock(the_mutex);
    the_queue.push_front(std::move(data));
  }
  bool empty() const {
    std::lock_guard<std::mutex> lock(the_mutex);
    return the_queue.empty();
  }
  bool try_pop(data_type& res) {
    std::lock_guard<std::mutex> lock(the_mutex);
    if (the_queue.empty()) {
      return false;
    }
    res = std::move(the_queue.front());
    the_queue.pop_front();
    return true;
  }
  bool try_steal(data_type& res) {
    std::lock_guard<std::mutex> lock(the_mutex);
    if (the_queue.empty()) {
      return false;
    }
    res = std::move(the_queue.back());
    the_queue.pop_back();
    return true;
  }
};

// namespace detail {
// thread_local work_stealing_queue* local_work_queue;
// thread_local unsigned int my_index;
// }


class work_stealing_thread_pool {
  typedef function_wrapper task_type;
  std::atomic_bool done;
  threadsafe_queue<task_type> pool_work_queue;
  std::vector<std::unique_ptr<work_stealing_queue> > queues;
  std::vector<std::thread> threads;
  join_threads joiner;
  static thread_local work_stealing_queue* local_work_queue;
  static thread_local unsigned int my_index;
  bool pop_task_from_local_queue(task_type& task) {
    return local_work_queue && local_work_queue->try_pop(task);
  }

  bool pop_task_from_pool_queue(task_type &task) {
    return pool_work_queue.try_pop(task);
  }

  bool pop_task_from_other_thread_queue(task_type &task) {
    for (unsigned int i = 0; i < queues.size(); ++i) {
      unsigned int const index = (my_index + i + 1) % queues.size();
      if (queues[index]->try_steal(task)) {
        return true;
      }
    }
    return false;
  }

public:
  void worker_thread(unsigned int my_index_) {
    my_index = my_index_;
    local_work_queue = queues[my_index].get();

    #ifdef __linux__
    nvtxNameOsThread(syscall(SYS_gettid), "threadpool");
    pthread_setname_np(pthread_self(), "threadpool");
    #endif

    while(!done) {
      run_pending_task();
    }
  }

  work_stealing_thread_pool(unsigned int thread_count):
    done(false), joiner(threads)
  {
    try {
      for (unsigned int i = 0; i <thread_count; ++i) {
        queues.push_back(std::make_unique<work_stealing_queue>());
      }
      for (unsigned int i = 0; i <thread_count; ++i) {
        threads.push_back(std::thread(&work_stealing_thread_pool::worker_thread, this, i));
      }
    } catch (...) {
      done = true;
      throw;
    }
  }

  ~work_stealing_thread_pool() {
    done = true;
  }

  template<typename FunctionType>
  std::future<typename std::result_of<FunctionType()>::type>
  submit(FunctionType f) {
    typedef typename std::result_of<FunctionType()>::type result_type;
    std::packaged_task<result_type()> task(f);
    std::future<result_type> res(task.get_future());
    if (local_work_queue) {
      local_work_queue->push(std::move(task));
    } else {
      pool_work_queue.push(std::move(task));
    }
    return res;
  }

  void run_pending_task() {
    task_type task;
    if (pop_task_from_local_queue(task) ||
        pop_task_from_pool_queue(task) ||
        // O(#threads). No good if threads never submit work to the
        // thread pool themselves...
        pop_task_from_other_thread_queue(task)) {
      task();
    } else {
      std::this_thread::yield();
    }
  }
};

} // namespace kaldi
