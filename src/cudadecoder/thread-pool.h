// cudadecoder/thread-pool.h
// Source:  https://github.com/progschj/ThreadPool
// Unmodified except for reformatting to Google style
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

#ifndef KALDI_CUDA_DECODER_THREAD_POOL_H_
#define KALDI_CUDA_DECODER_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool {
public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  template <class F, class... Args>
  auto enqueue_high_priority(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

 private:
  template <class F, class... Args>
  auto enqueue(bool insert_front, F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::deque<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;

  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

	{
	std::unique_lock<std::mutex> lock(this->queue_mutex);
        this->condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });
        if (this->stop && this->tasks.empty()) return;
        if (!tasks.empty()) {
          task = std::move(this->tasks.front());
          this->tasks.pop_front();
        }
	}

        task();
      }
    });
}

// add new work item to the pool : normal priority
// executed in FIFO order
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  return enqueue(false, std::forward<F>(f), std::forward<Args>(args)...);
}

// add new work item to the pool : high priority
// this task will be put directly at the front of the task queue
template <class F, class... Args>
auto ThreadPool::enqueue_high_priority(F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  return enqueue(true, std::forward<F>(f), std::forward<Args>(args)...);
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(bool insert_front, F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    if (insert_front)
      tasks.emplace_front([task]() { (*task)(); });
    else
      tasks.emplace_back([task]() { (*task)(); });
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
  for (std::thread &worker : workers)
    worker.join();
}

#endif  // KALDI_CUDA_DECODER_THREAD_POOL_H_
