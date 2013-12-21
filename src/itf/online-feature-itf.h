// itf/online-feature-itf.h

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_ITF_ONLINE_FEATURE_ITF_H_
#define KALDI_ITF_ONLINE_FEATURE_ITF_H_ 1
#include "base/kaldi-common.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{


/// OnlineFeatureInterface is an interface for online feature processing, also
/// usable in the offline setting.  This is for use in the online2/ directory,
/// and it supersedes the interface in ../online/online-feat-input.h.  We have a
/// slighty different model that puts more control in the hands of the calling
/// thread, and won't involve waiting on semaphores in the decoding thread.
///
/// This interface only specifies how the object *outputs* the features.
/// How it obtains the features, e.g. from a previous object or objects of type
/// OnlineFeatureInterface, is not specified in the interface and you will
/// likely define constructors or methods that are specific to the derived type
/// which will take care of that.
///
/// Notice the Recompute() function.  If you call this it signals the class that
/// it can, if it wants, recompute the features in a better way, for example
/// replacing online cepstral mean normalization with whole-file cepstral mean
/// normalization.  The likely usage of this from the calling code is to
/// decode the file and make lattices; call Recompute(); and then get all
/// the features again to rescore the lattice.

class OnlineFeatureInterface {
 public:
  virtual int32 Dim() const; /// returns the feature dimension.

  /// Calling Recompute() will signal the class that it may recompute
  /// all its features if it wants (e.g. convert online to batch-mode CMN,
  /// and the same for any algorithms, such as pitch extraction, which are
  /// not purely frame-to-frame).
  /// If two calls to GetFeature() are separated by a call to Recompute(),
  /// there is no expetation that the two calls will return the same
  /// feature.
  virtual void Recompute();

  /// Returns true if this is the last frame.  Frames are zero-based, so the
  /// first frame is zero.  IsLastFrame(-1) will return false, unless the file
  /// is empty (which is a case that I'm not sure all the code will handle, so
  /// be careful).  Caution: the behavior of this function in an online setting
  /// is being changed somewhat.  In future it may return false in cases where
  /// we haven't yet decided to terminate decoding, but later true if we decide
  /// to terminate decoding.  The plan in future is to rely more on
  /// NumFramesReady(), and in future, IsLastFrame() would always return false
  /// in an online-decoding setting, and would only return true in a
  /// decoding-from-matrix setting where we want to allow the last delta or LDA
  /// features to be flushed out for compatibility with the baseline setup.
  virtual bool IsLastFrame(int32 frame) = 0;


  /// Returns the number of frames that are currently ready to be processed.
  virtual int32 NumFramesReady();

  
  /// Gets the feature for this frame.  Before calling this for a given frame,
  /// it's assumed that you have already called IsLastFrame(frame - 1) and it
  /// returned false, or [preferably] you called FrameIsReady(frame) and it returned
  /// true.  Otherwise it may crash.
  virtual void GetFeature(int32 frame, VectorBase<BaseFloat> *feat);  
};

/// @}
}  // namespace Kaldi

#endif  // KALDI_ITF_ONLINE_FEATURE_ITF_H_
