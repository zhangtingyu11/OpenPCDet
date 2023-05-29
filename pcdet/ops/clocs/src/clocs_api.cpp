#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "clocs.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clocs_comput_iou", &clocs_compute_iou_gpu, "compute clocs iou");
}
