#include <cudadecoder/thread-pool-cia.h>

namespace kaldi {
thread_local work_stealing_queue* work_stealing_thread_pool::local_work_queue;
thread_local unsigned int work_stealing_thread_pool::my_index;
}
