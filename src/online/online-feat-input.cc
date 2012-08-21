// online/online-feat-input.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#include "online/online-feat-input.h"

namespace kaldi {

bool
OnlineCmvnInput::Compute(Matrix<BaseFloat> *output, uint32 *timeout) {
  in_matrix_.Resize(output->NumRows(), output->NumCols(), kUndefined);
  bool more_data = input_->Compute(&in_matrix_, timeout);
  cmvn_.ApplyCmvn(in_matrix_, output);
  return more_data;
}


OnlineUdpInput::OnlineUdpInput(int32 port) {
  server_addr_.sin_family = AF_INET; // IPv4
  server_addr_.sin_addr.s_addr = INADDR_ANY; // listen on all interfaces
  server_addr_.sin_port = htons(port);
  sock_desc_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock_desc_ == -1)
    KALDI_ERR << "socket() call failed!";
  int32 rcvbuf_size = 30000;
  if (setsockopt(sock_desc_, SOL_SOCKET, SO_RCVBUF,
                 &rcvbuf_size, sizeof(rcvbuf_size)) == -1)
      KALDI_ERR << "setsockopt() failed to set receive buffer size!";
  if (bind(sock_desc_,
           reinterpret_cast<sockaddr*>(&server_addr_),
           sizeof(server_addr_)) == -1)
    KALDI_ERR << "bind() call failed!";
}


bool
OnlineUdpInput::Compute(Matrix<BaseFloat> *output, uint32 *timeout) {
  KALDI_ASSERT(timeout == 0 &&
               "Timeout parameter currently not supported by OnlineUdpInput!");
  char buf[65535];
  socklen_t caddr_len = sizeof(client_addr_);
  ssize_t nrecv = recvfrom(sock_desc_, buf, sizeof(buf), 0,
                           reinterpret_cast<sockaddr*>(&client_addr_),
                           &caddr_len);
  if (nrecv == -1) {
    KALDI_WARN << "recvfrom() call error!";
    output->Resize(0, 0);
    return false;
  }
  std::stringstream ss(std::stringstream::in | std::stringstream::out);
  ss.write(buf, nrecv);
  output->Read(ss, true);
  return true;
}


OnlineLdaInput::OnlineLdaInput(OnlineFeatInputItf *input, const uint32 feat_dim,
                               const Matrix<BaseFloat> &transform,
                               const uint32 left_context,
                               const uint32 right_context)
    : input_(input), feat_dim_(feat_dim),
      transform_(transform), window_size_(left_context + right_context + 1),
      window_center_(left_context),
      window_pos_(0),
      trans_rows_(transform.NumRows()) {}


void OnlineLdaInput::InitFeatWindow() {
  feat_window_.Resize(window_size_, feat_dim_, kUndefined);
  int32 i;
  for (i = 0; i <= window_center_; ++i)
    feat_window_.CopyRowFromVec(feat_in_.Row(0), i);
  for (; i < window_size_; ++i)
    feat_window_.CopyRowFromVec(feat_in_.Row(i - window_center_ - 1), i);
  window_pos_ = 0;
}


bool OnlineLdaInput::Compute(Matrix<BaseFloat> *output, uint32 *timeout) {
  // Receive raw features from the inferior
  MatrixIndexT nrows = output->NumRows();
  MatrixIndexT ncols = output->NumCols();
  KALDI_ASSERT(ncols == trans_rows_ && "Invalid feature dimensionality!");
  feat_in_.Resize(nrows, feat_dim_, kUndefined);
  bool more_data = input_->Compute(&feat_in_, timeout);
  if (feat_in_.NumRows() != nrows) {
    KALDI_VLOG(3) << "Fewer than requested features received!";
    if (feat_in_.NumRows() == 0 ||
        (feat_in_.NumRows() < window_size_ && feat_window_.NumRows() == 0)) {
      output->Resize(0, 0);
      return more_data;
    }
  }

  // Prepare the splice window
  int32 cur_feat = 0;
  if (feat_window_.NumRows() == 0) {
    InitFeatWindow();
    cur_feat = window_size_ - window_center_ - 1;
  }
  spliced_feats_.Resize(feat_in_.NumRows() - cur_feat,
                        feat_dim_ * window_size_,
                        kUndefined);
  output->Resize(feat_in_.NumRows() - cur_feat, trans_rows_, kUndefined);
  int32 sfi = 0; // index of the current row in spliced_feats_
  for (; cur_feat < feat_in_.NumRows(); ++cur_feat, ++sfi) {
    // append at the tail of the window and advance the effective window position
    feat_window_.CopyRowFromVec(feat_in_.Row(cur_feat), window_pos_);
    window_pos_ = (++window_pos_) % window_size_;
    // splice the feature vectors in the current feature window
    int32 fwi = window_pos_; // the oldest vector in feature window
    SubVector<BaseFloat> spliced_row(spliced_feats_, sfi);
    for (int32 i = 0; i < window_size_; ++i) {
      SubVector<BaseFloat> dst(spliced_row, i*feat_dim_, feat_dim_);
      dst.CopyRowFromMat(feat_window_, fwi);
      fwi = (fwi + 1) % window_size_;
    }
  }
  output->AddMatMat(1.0, spliced_feats_, kNoTrans, transform_, kTrans, 0.0);
  return more_data;
}


OnlineDeltaInput::OnlineDeltaInput(OnlineFeatInputItf *input, uint32 feat_dim,
                                   uint32 order, uint32 window)
    : input_(input), feat_dim_(feat_dim), order_(order),
      window_size_(2*window*order + 1), window_center_(window*order),
      delta_(DeltaFeaturesOptions(order, window)) { }


void
OnlineDeltaInput::InitFeatWindow() {
  feat_window_.Resize(window_size_, feat_dim_, kUndefined);
  int32 i;
  for (i = 0; i <= window_center_; ++i)
    feat_window_.CopyRowFromVec(feat_in_.Row(0), i);
  for (; i < window_size_; ++i)
    feat_window_.CopyRowFromVec(feat_in_.Row(i - window_center_ - 1), i);
}


bool
OnlineDeltaInput::Compute(Matrix<BaseFloat> *output, uint32 *timeout) {
  // Receive raw features from the inferior
  MatrixIndexT nrows = output->NumRows();
  MatrixIndexT ncols = output->NumCols();
  uint32 out_dim = feat_dim_ * (order_ + 1);
  KALDI_ASSERT(ncols == out_dim && "Invalid feature dimensionality!");
  feat_in_.Resize(nrows, feat_dim_, kUndefined);
  bool more_data = input_->Compute(&feat_in_, timeout);
  if (feat_in_.NumRows() != nrows) {
    KALDI_VLOG(3) << "Fewer than requested features received!";
    if (feat_in_.NumRows() == 0 ||
        (feat_in_.NumRows() < window_size_ && feat_window_.NumRows() == 0)) {
      output->Resize(0, 0);
      return more_data;
    }
  }

  // Prepare the delta window
  int32 cur_feat = 0;
  if (feat_window_.NumRows() == 0) {
    InitFeatWindow();
    cur_feat = window_size_ - window_center_ - 1;
  }

  // Calculate and output features with deltas added
  output->Resize(feat_in_.NumRows() - cur_feat, out_dim);
  int32 ofi = 0; // the index of the output feature
  for (; cur_feat < feat_in_.NumRows(); ++cur_feat, ++ofi) {
    // Update(shift up) the feature window
    // ToDo(vdp): this looks quite inefficient - maybe use a matrix, say 10
    // times larger than the feature window and perform a copy once per
    // every 9*window_size_ feature vectors
    for (int32 i = 0; i < feat_window_.NumRows() - 1; ++i)
      feat_window_.CopyRowFromVec(feat_window_.Row(i+1), i);
    feat_window_.CopyRowFromVec(feat_in_.Row(cur_feat), window_size_ - 1);

    SubVector<BaseFloat> out_vec(*output, ofi);
    delta_.Process(feat_window_, window_center_, &out_vec);
  }
  return more_data;
} // OnlineDeltaInput::Compute()

} // namespace kaldi

