// cudadecoder/cuda-decoder.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary, Daniel Galvez
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

#ifndef KALDI_CUDADECODER_THREAD_POOL_LIGHT_H_
#define KALDI_CUDADECODER_THREAD_POOL_LIGHT_H_

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace kaldi {
namespace cuda_decoder {

constexpr double kSleepForWorkAvailable = 1e-3;
constexpr double kSleepForWorkerAvailable = 1e-3;

struct ThreadPoolLightTask {
  void (*func_ptr)(void *, uint64_t, void *);
  void *obj_ptr;
  uint64_t arg1;
  void *arg2;
};

template <int QUEUE_SIZE>
// Single producer, multiple consumer
class ThreadPoolLightSPMCQueue {
  static constexpr unsigned int QUEUE_MASK = QUEUE_SIZE - 1;
  std::vector<ThreadPoolLightTask> tasks_{QUEUE_SIZE};
  std::atomic<int> back_{0};
  std::atomic<int> front_{0};
  static int inc(int curr) { return ((curr + 1) & QUEUE_MASK); }

 public:
  ThreadPoolLightSPMCQueue() {
    KALDI_COMPILE_TIME_ASSERT(QUEUE_SIZE > 1);
    constexpr bool is_power_of_2 = ((QUEUE_SIZE & (QUEUE_SIZE - 1)) == 0);
    KALDI_COMPILE_TIME_ASSERT(is_power_of_2);  // validity of QUEUE_MASK
  }

  bool TryPush(const ThreadPoolLightTask &task) {
    int back = back_.load(std::memory_order_relaxed);
    int next = inc(back);
    if (next == front_.load(std::memory_order_acquire)) {
      return false;  // queue is full
    }
    tasks_[back] = task;
    back_.store(next, std::memory_order_release);

    return true;
  }

  bool TryPop(ThreadPoolLightTask *front_task) {
    while (true) {
      int front = front_.load(std::memory_order_relaxed);
      if (front == back_.load(std::memory_order_acquire)) {
        return false;  // queue is empty
      }
      *front_task = tasks_[front];
      if (front_.compare_exchange_weak(front, inc(front),
                                       std::memory_order_release)) {
        return true;
      }
    }
  }
};

class ThreadPoolLightWorker final {
  // Multi consumer queue, because worker can steal work
  ThreadPoolLightSPMCQueue<512> queue_;
  // If this thread has no more work to do, it will try to steal work from
  // other
  std::thread thread_;
  volatile bool run_thread_;
  ThreadPoolLightTask curr_task_;
  std::weak_ptr<ThreadPoolLightWorker> other_;

  void Work() {
    while (run_thread_) {
      bool got_task = queue_.TryPop(&curr_task_);
      if (!got_task) {
        if (auto other_sp = other_.lock()) {
          got_task = other_sp->TrySteal(&curr_task_);
        }
      }
      if (got_task) {
        // Not calling func_ptr as a member function,
        // because we need to specialize the arguments
        // anyway (we may want to ignore arg2, for
        // instance) Using a wrapper func
        (curr_task_.func_ptr)(curr_task_.obj_ptr, curr_task_.arg1,
                              curr_task_.arg2);
      } else {
        Sleep(kSleepForWorkAvailable);  // TODO
      }
    }
  }

  // Another worker can steal a task from this queue
  // This is done so that a very long task computed by one thread does not
  // hold the entire threadpool to complete a time-sensitive task
  bool TrySteal(ThreadPoolLightTask *task) { return queue_.TryPop(task); }

 public:
  ThreadPoolLightWorker() : run_thread_(true), other_() {}
  ~ThreadPoolLightWorker() {
     KALDI_ASSERT(!queue_.TryPop(&curr_task_));
  }
  bool TryPush(const ThreadPoolLightTask &task) {
    return queue_.TryPush(task);
  }
  void SetOtherWorkerToStealFrom(
      const std::shared_ptr<ThreadPoolLightWorker>& other) {
    other_ = other;
  }
  void Start() {
    KALDI_ASSERT("Please call SetOtherWorkerToStealFrom() first" &&
                 !other_.expired());
    thread_ = std::thread(&ThreadPoolLightWorker::Work, this);
  }
  void Stop() {
    run_thread_ = false;
    thread_.join();
    other_.reset();
  }
};

class ThreadPoolLight {
  std::vector<std::shared_ptr<ThreadPoolLightWorker>> workers_;
  int curr_iworker_;  // next call on tryPush will post work on this
                      // worker
 public:
  ThreadPoolLight(int32 nworkers = std::thread::hardware_concurrency())
      : workers_(nworkers), curr_iworker_(0) {
    KALDI_ASSERT(nworkers > 1);
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i] = std::make_shared<ThreadPoolLightWorker>();
    }
    for (size_t i = 0; i < workers_.size(); ++i) {
      int iother = (i + nworkers / 2) % nworkers;
      workers_[i]->SetOtherWorkerToStealFrom(workers_[iother]);
      workers_[i]->Start();
    }
  }

  ~ThreadPoolLight() {
    for (auto& wkr : workers_) wkr->Stop();
  }

  bool TryPush(const ThreadPoolLightTask &task) {
    if (!workers_[curr_iworker_]->TryPush(task)) return false;
    ++curr_iworker_;
    if (curr_iworker_ == workers_.size()) curr_iworker_ = 0;
    return true;
  }

  void Push(const ThreadPoolLightTask &task) {
    // Could try another curr_iworker_
    while (!TryPush(task)) {
      Sleep(kSleepForWorkerAvailable);
    }
  }
};

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDADECODER_THREAD_POOL_LIGHT_H_
