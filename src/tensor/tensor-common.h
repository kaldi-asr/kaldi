/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {



enum {
  kCpuDevice = 0,
  kCudaDevice = 1
} DeviceType;


// We may later add a device number (like which GPU we are using),
// once we support multiple GPUs.
struct Device {
  DeviceType device_type;
  // operator ==, probably, maybe constructors.
};


enum DataType {
  // We will of course later extend this with many more types, including
  // integer types and half-precision floats.
  kFloatDtype = 0,
  kDoubleDtype = 1
};



/// Enumeration that says what strides we should choose when allocating
/// A Tensor.
enum StridePolicy {
  kCopyStrides,  // means: copy the strides from the source Tensor, preserving
                 //  their signs and relative ordering (but filling in gaps if
                 //  the source Tensor's data was not contiguous.
  kCstrides      // means: strides for dimensions that are != 1 are ordered from
                 // greatest to smallest as in a "C" array.  Per our policy,
                 // any dimension that is 1 will have a zero stride.
};

/// Enumeration that says whether to zero a freshly initialized Tensor.
enum InitializePolicy {
  kZeroData,
  kUninitialized
};


#define KALDI_TENSOR_MAX_DIM 5


};
