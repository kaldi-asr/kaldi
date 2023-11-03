// cudadecoder/thread-pool-cia.cc
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

#include <cudadecoder/thread-pool-cia.h>

namespace kaldi {
thread_local work_stealing_queue* work_stealing_thread_pool::local_work_queue;
thread_local unsigned int work_stealing_thread_pool::my_index;
}
