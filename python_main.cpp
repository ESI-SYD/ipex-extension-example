#include <torch/extension.h>
#include <ipex.h>
#include <vector>
#include "softmax.h"

sycl::queue get_current_sycl_queue() {
    // submit kernel
    c10::impl::VirtualGuardImpl impl(at::DeviceType::XPU);
    c10::Stream stream = impl.getStream(impl.getDevice());

    return xpu::get_queue_from_stream(stream);
}

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)

at::Tensor softmax_shape0(const at::Tensor& input, const int64_t dim) {
  CHECK_INPUT(input);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<mat1_4096x2048_bf16_cfg0>(input.data_ptr(), output.data_ptr(), queue);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("softmax_shape0", &softmax_shape0, "softmax forward (XeTLA)");
//m.def("softmax", &softmax_shape0, "softmax forward (XeTLA)");
}