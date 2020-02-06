// cudafeature/feature-online-batched-spectral-cuda-kernels.h
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens, Levi Barnes
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

#ifndef KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_SPECTRAL_CUDA_KERNELS_H_
#define KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_SPECTRAL_CUDA_KERNELS_H_

#include "cudafeat/lane-desc.h"

namespace kaldi {

void cuda_power_spectrum(int32_t max_chunk_frames, int32_t num_lanes,
                         int row_length, const float *A_in, int32_t ldi,
                         float *A_out, int32_t ldo, bool use_power);

void cuda_mel_banks_compute(const LaneDesc *lanes, int32_t n_lanes,
                            int32_t max_chunk_frames, int32_t num_bins,
                            float energy_floor, int32_t *offsets,
                            int32_t *sizes, float **vecs, const float *feats,
                            int32_t ldf, float *mels, int32_t ldm,
                            bool use_log);

void cuda_apply_lifter_and_floor_energy(const LaneDesc *lanes,
                                        int32_t num_lanes,
                                        int32_t max_chunk_frames, int num_cols,
                                        float cepstral_lifter, bool use_energy,
                                        float energy_floor, float *log_energy,
                                        int32_t ldl, float *lifter_coeffs,
                                        float *features, int32_t ldf);

void cuda_process_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int frame_length,
                         float dither, float energy_floor,
                         bool remove_dc_offset, float preemph_coeff,
                         bool need_raw_log_energy, float *log_energy_pre_window,
                         int32_t lde, const float *windowing,
                         float *tmp_windows, int32_t ldt, float *windows,
                         int32_t ldw);

void cuda_extract_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int32_t frame_shift,
                         int32_t frame_length, int32_t frame_length_padded,
                         bool snip_edges, const float *wave, int32_t ldw,
                         float *windows, int32_t window_size, int32_t wlda,
                         float *stash, int32_t ssize, int32_t lds);

void cuda_dot_log(int32_t max_chunk_frames, int32_t num_lanes,
                  int32_t frame_length, float *signal_frame, int32_t lds,
                  float *signal_log_energy, int32_t lde);

void cuda_update_stash(const LaneDesc *lanes, int32_t num_lanes,
                       const float *wave, int32_t ldw, float *stash,
                       int32_t num_stash, int32_t lds);

}  // namespace kaldi
#endif
