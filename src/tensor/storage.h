#include "tensor/tensor-common.h"

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {




// 'Storage' contains a single allocated region (on CPU or GPU, according
// to 'device').
struct Storage {
  void *data;
  size_t num_bytes;
  Device device;

  // Note: will throw if allocation fails (for now).
  Storage(Device device, size_t num_bytes);

  // Destructor deallocates 'data'.  For now there is no
  // concept of a custom allocator or an allocator object, we just use our CuDevice stuff for cuda
  // allocation and posix_memalign for CPU allocation (obviously we need
  // to make sure 'data' is aligned in most specific way we might need).
  // in future we might choose
  // to add that.
  ~Storage();
};





};
