// cudadecoder/thread-pool.h
// Source:  https://github.com/progschj/ThreadPool
// Modified to add a priority queue
// Ubtained under this license:
/*
Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/

//
// Important: This file is deprecated and will be removed in a future release
//

#ifndef KALDI_CUDA_DECODER_DEPRECATED_THREAD_POOL_H_
#define KALDI_CUDA_DECODER_DEPRECATED_THREAD_POOL_H_

#include <climits>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kaldi {
namespace cuda_decoder {

// C++ indexes enum 0,1,2...
enum [[deprecated]] ThreadPoolPriority {
  THREAD_POOL_LOW_PRIORITY,
  THREAD_POOL_NORMAL_PRIORITY,
  THREAD_POOL_HIGH_PRIORITY
};

class [[deprecated]] ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(ThreadPoolPriority priority, F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  template <class F, class... Args>
  auto enqueue(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  struct Task {
    std::function<void()> func;
    // Ordered first by priority, then FIFO order
    // tasks created first will have a higher
    // priority_with_fifo.second
    std::pair<ThreadPoolPriority, long long> priority_with_fifo;
  };
  friend bool operator<(const ThreadPool::Task &lhs,
                        const ThreadPool::Task &rhs);

  std::priority_queue<Task> tasks;
  long long task_counter;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;

  bool stop;
};

inline bool operator<(const ThreadPool::Task &lhs,
                      const ThreadPool::Task &rhs) {
  return lhs.priority_with_fifo < rhs.priority_with_fifo;
}

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    : task_counter(LONG_MAX), stop(false) {
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        Task task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty()) return;
          if (!tasks.empty()) {
            task = std::move(this->tasks.top());
            this->tasks.pop();
          }
        }
        task.func();
      }
    });
}

// add new work item to the pool : normal priority
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  return enqueue(THREAD_POOL_NORMAL_PRIORITY, std::forward<F>(f),
                 std::forward<Args>(args)...);
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(ThreadPoolPriority priority, F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto func = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = func->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
    Task task;
    task.func = [func]() { (*func)(); };
    long long task_fifo_id = task_counter--;
    // The following if will temporarly break the FIFO order
    // (leading to a perf drop for a few seconds)
    // But it should trigger in ~50 million years
    if (task_counter == 0) task_counter = LONG_MAX;
    task.priority_with_fifo = {priority, task_fifo_id};
    tasks.push(std::move(task));
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) worker.join();
}

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // KALDI_CUDA_DECODER_THREAD_POOL_H_
