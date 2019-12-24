// pybind/dlpack/dlpack_deleter.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "dlpack/dlpack_deleter.h"
#include <iostream>

namespace kaldi {

/*
 * To check that the deleter is indeed called by the consumer of the PyCapsule,
 * you can print a log message inside this deleter.
 *
 * Take PyTorch as an example, for the following test Python code

```python
from torch.utils.dlpack import from_dlpack
import kaldi

v = kaldi.FloatVector(3)

tensor = from_dlpack(v.to_dlpack())
print('begin to delete tensor')
del tensor
print('you should see the deleter has been called')
```

If you put:
```cpp
  std::cout << "deleter is called!" << std::endl;
```
inside the following function, you should see the following output
for the above python program:

```
begin to delete tensor
deleter is called!
you should see the deleter has been called
```
 *
 */

void DLManagedTensorDeleter(struct DLManagedTensor* dl_managed_tensor) {
  // data is shared, so do NOT free data

  // `shape` is created with `new[]`
  delete[] dl_managed_tensor->dl_tensor.shape;

  // `strides` is created with `new[]`
  delete[] dl_managed_tensor->dl_tensor.strides;

  // now delete the DLManagedTensor which is created with `new`
  delete dl_managed_tensor;
  std::cout << "deleter is called!" << std::endl;
}

}  // namespace kaldi
