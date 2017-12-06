// online2/online2-nnet3-latgen-i2x-wrapper.h
#ifndef KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_
#define KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_
#include <stdint.h>
namespace kaldi {
class DecoderFactory;
class Decoder; 
} // namespace kaldi

using kaldi::DecoderFactory;
using kaldi::Decoder;
/*
   Creates a decoder factory.
   Called only once during the lifetime.
   mmaps extremely heavy resources like WFST and AM (up to 20 GB).
   Calling it more than once is likely to cause OOM!
   Returns a handle to a decoder factory,
   which will create light-weighted decoder objects (one per session).
   Returns nullptr on failure.
*/
DecoderFactory *InitDecoderFactory(const std::string &resource_dir);

// Creates a decoder object.
// Returns nullptr on failure.
Decoder *StartDecodingSession(const DecoderFactory *);

// Feed PCM UI16 data into the decoder.
// Returns 0 on success, error code otherwise.
// If called with length == 0, the decoder is finalized and no further calls are allowed.
int32_t FeedChunk(Decoder *, uint16_t *data, size_t length);

// Puts the final recognition result in the string passed by pointer.
// Frees the resources and destroys the recognition session.
// Returns 0 on success, error code otherwise.
int32_t GetResultAndFinalize(Decoder *decoder, std::string *result);
#endif // KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_

