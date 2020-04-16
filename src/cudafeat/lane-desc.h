// cudafeat/lane-desc.h
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
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

#ifndef CUDAFEAT_LANE_DESC_H_
#define CUDAFEAT_LANE_DESC_H_

#include <cstdint>

namespace kaldi {

typedef int32_t ChannelId;

// The description for a single channel.
// A vector of these will be passed into components to
// control which channels are executed.
struct LaneDesc {
  ChannelId channel;

  // number of samples in this chunk
  int32_t num_chunk_samples;

  // current sample for this chunk
  int32_t current_sample;

  // number of frames in this chunk.
  int32_t num_chunk_frames;

  // current frame for this chunk
  int32_t current_frame;

  // is this the last chunk
  int32_t last;

  // is this the first chunk
  int32_t first;
};

}  // namespace kaldi
#endif
