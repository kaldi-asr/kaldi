// feat/rir-generator.cc

// Copyright 2018 Jian Wu

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

#include "feat/rir-generator.h"

namespace kaldi {

void RirGenerator::ComputeDerived() {
  if (!str_to_pattern_.count(opts_.microphone_type))
    KALDI_ERR << "Unknown option values: --microphone-type="
              << opts_.microphone_type;
  KALDI_ASSERT(frequency_ >= 1);
  KALDI_ASSERT(velocity_ >= 1);
  KALDI_ASSERT(order_ >= -1);

  // Process room topo
  KALDI_ASSERT(opts_.room_topo != "" &&
               "Options --room-topo is not configured");
  std::vector<BaseFloat> topo;
  KALDI_ASSERT(SplitStringToFloats(opts_.room_topo, ",", false, &topo));
  room_dim_ = topo.size();
  KALDI_ASSERT(room_dim_ == 3);
  room_topo_.CopyFromVector(topo);

  // Process source location
  KALDI_ASSERT(opts_.source_location != "" &&
               "Options --source-location is not configured");
  std::vector<BaseFloat> loc;
  KALDI_ASSERT(SplitStringToFloats(opts_.source_location, ",", false, &loc));
  source_location_.CopyFromVector(loc);

  // Process receiver_location
  KALDI_ASSERT(opts_.receiver_location != "" &&
               "Options --receiver-location is not configured");
  std::vector<std::string> mics;
  SplitStringToVector(opts_.receiver_location, ";", false, &mics);
  num_mics_ = mics.size();
  receiver_location_.resize(num_mics_);
  for (int32 i = 0; i < num_mics_; i++) {
    KALDI_ASSERT(SplitStringToFloats(mics[i], ",", false, &loc));
    receiver_location_[i].CopyFromVector(loc);
  }

  // Process angle
  std::vector<BaseFloat> angle_tmp;
  if (opts_.orientation == "") {
    for (int32 i = 0; i <= 1; i++) angle_tmp.push_back(0.0);
  } else {
    KALDI_ASSERT(
        SplitStringToFloats(opts_.orientation, ",", false, &angle_tmp));
  }
  if (angle_tmp.size() == 1) angle_tmp.push_back(0.0);
  KALDI_ASSERT(angle_tmp.size() == 2);
  angle_.swap(angle_tmp);

  // Process beta
  std::vector<BaseFloat> beta_tmp;
  KALDI_ASSERT(opts_.beta != "" && "Options --beta is not configured");
  KALDI_ASSERT(SplitStringToFloats(opts_.beta, ",", false, &beta_tmp));
  // dim of beta: 1 means RT60
  //              6 means reflection coefficients for each surface 
  KALDI_ASSERT(beta_tmp.size() == 1 || beta_tmp.size() == 6);
  if (beta_tmp.size() == 1) {
    // beta_tmp[0] is T60
    revb_time_ = beta_tmp[0];
    BaseFloat V = room_topo_.V(), S = room_topo_.S();
    beta_tmp.resize(6);
    if (revb_time_ != 0) {
      BaseFloat alfa = 24 * V * Log(10.0) / (velocity_ * S * revb_time_);
      if (alfa > 1)
        KALDI_ERR << alfa << " > 1: The reflection coefficients cannot be "
                             "calculated using the current"
                  << " room parameters, i.e. room size and reverberation time.";
      for (int32 i = 0; i < 6; i++) beta_tmp[i] = std::sqrt(1 - alfa);
    } else {
      for (int32 i = 0; i < 6; i++) beta_tmp[i] = 0;
    }
  } else {
    // compute from sabine formula
    revb_time_ = Sabine(room_topo_, beta_tmp, velocity_);
    revb_time_ = std::max(0.128f, revb_time_);
  }
  beta_.swap(beta_tmp);
  KALDI_ASSERT(beta_.size() == 6);

  // Process number of samples
  // if non-positive, compute from T60
  if (!num_samples_) {
    num_samples_ = static_cast<int32>(revb_time_ * frequency_);
  }
  KALDI_ASSERT(num_samples_ && "Invalid number of samples");
}

void RirGenerator::GenerateRir(Matrix<BaseFloat> *rir) {
  rir->Resize(num_mics_, num_samples_);
  const BaseFloat cts = velocity_ / frequency_;
  Point3D S(source_location_), T(room_topo_);
  Point3D Y;
  Point3D Rp_plus_Rm, Rm, Refl;

  S.Scale(1.0 / cts);
  T.Scale(1.0 / cts);

  BaseFloat dist, fdist, gain;
  // same configure as RIR-generator
  int32 Tw = 2 * static_cast<int32>(0.004 * frequency_ + 0.5);
  BaseFloat W = 2 * M_PI * 100 / frequency_;
  BaseFloat R1 = exp(-W), B1 = 2 * R1 * cos(W), B2 = -R1 * R1, A1 = -1 - R1;

  for (int32 m = 0; m < num_mics_; m++) {
    Point3D R(receiver_location_[m]);
    R.Scale(1.0 / cts);

    int32 nx = static_cast<int32>(ceil(num_samples_ / (2 * T.x)));
    int32 ny = static_cast<int32>(ceil(num_samples_ / (2 * T.y)));
    int32 nz = static_cast<int32>(ceil(num_samples_ / (2 * T.z)));

    for (int32 x = -nx; x <= nx; x++) {
      Rm.x = 2 * x * T.x;

      for (int32 y = -ny; y <= ny; y++) {
        Rm.y = 2 * y * T.y;

        for (int32 z = -nz; z <= nz; z++) {
          Rm.z = 2 * z * T.z;

          for (int32 q = 0; q <= 1; q++) {
            Rp_plus_Rm.x = (1 - 2 * q) * S.x - R.x + Rm.x;
            Refl.x = pow(beta_[0], abs(x - q)) * pow(beta_[1], abs(x));

            for (int32 j = 0; j <= 1; j++) {
              Rp_plus_Rm.y = (1 - 2 * j) * S.y - R.y + Rm.y;
              Refl.y = pow(beta_[2], abs(y - j)) * pow(beta_[3], abs(y));

              for (int32 k = 0; k <= 1; k++) {
                Rp_plus_Rm.z = (1 - 2 * k) * S.z - R.z + Rm.z;
                Refl.z = pow(beta_[4], abs(z - k)) * pow(beta_[5], abs(z));

                dist = Rp_plus_Rm.L2Norm();
                if (abs(2 * x - q) + abs(2 * y - j) + abs(2 * z - k) <=
                        order_ ||
                    order_ == -1) {
                  fdist = floor(dist);

                  if (fdist < num_samples_) {
                    int32 pos = static_cast<int32>(fdist - (Tw / 2) + 1);
                    gain = MicrophoneSim(Rp_plus_Rm) * Refl.V() /
                           (4 * M_PI * dist * cts);

                    for (int32 n = 0; n < Tw; n++) {
                      if (pos + n >= 0 && pos + n < num_samples_)
                        (*rir)(m, pos + n) +=
                            gain * (0.5 * (1 - cos(2 * M_PI * ((n + 1 - 
                            (dist - fdist)) / Tw))) *
                            Sinc(M_PI * (n + 1 - (dist - fdist) - (Tw / 2))));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (hp_filter_) {
      Y.Reset();
      BaseFloat X0;
      for (int32 i = 0; i < num_samples_; i++) {
        X0 = (*rir)(m, i);
        Y.z = Y.y;
        Y.y = Y.x;
        Y.x = B1 * Y.y + B2 * Y.z + X0;
        (*rir)(m, i) = Y.x + A1 * Y.y + R1 * Y.z;
      }
    }
  }
}

BaseFloat RirGenerator::MicrophoneSim(const Point3D &p) {
  BaseFloat rho = 0;
  switch (str_to_pattern_[opts_.microphone_type]) {
    case kBidirectional:
      rho = 0;
      break;
    case kHypercardioid:
      rho = 0.25;
      break;
    case kCardioid:
      rho = 0.5;
      break;
    case kSubcardioid:
      rho = 0.75;
      break;
    case kOmnidirectional:
      rho = 1;
      break;
  }
  if (rho == 1) {
    return 1;
  } else {
    BaseFloat theta = acos(p.z / p.L2Norm());
    BaseFloat phi = atan2(p.y, p.x);
    BaseFloat gain =
        sin(M_PI / 2 - angle_[1]) * sin(theta) * cos(angle_[0] - phi) +
        cos(M_PI / 2 - angle_[1]) * cos(theta);
    return rho + (1 - rho) * gain;
  }
}

std::string RirGenerator::Report() {
  std::ostringstream oss;
  oss << "RirGenerator Configures: " << std::endl;
  oss << "-- Sound Velocity: " << velocity_ << std::endl;
  oss << "-- Sample Frequency:  " << frequency_ << std::endl;
  oss << "-- Number of Samples: " << num_samples_ << std::endl;
  oss << "-- Order/Room Dim: " << order_ << "/" << room_dim_ << std::endl;
  oss << "-- PolarPattern: " << opts_.microphone_type << std::endl;
  oss << "-- Reverberation Time: " << revb_time_ << std::endl;
  oss << "-- Source Location: (" << source_location_.x << ", "
      << source_location_.y << ", " << room_topo_.z << ")" << std::endl;
  oss << "-- Room Topology: (" << room_topo_.x << ", " << room_topo_.y << ", "
      << room_topo_.z << ")" << std::endl;
  oss << "-- Angle: [ ";
  std::copy(angle_.begin(), angle_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  oss << "-- Beta Vector: [ ";
  std::copy(beta_.begin(), beta_.end(), std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  oss << "-- Reciver Locations: ";
  for (int32 i = 0; i < receiver_location_.size(); i++) {
    oss << "(" << receiver_location_[i].x << ", " << receiver_location_[i].y
        << ", " << receiver_location_[i].z << ") ";
  }
  oss << std::endl;
  return oss.str();
}

}  // namespace kaldi
