import torch
import intel_extension_for_pytorch
import xetla_kernel

a=torch.empty((96,2048,2048), device='xpu', dtype=torch.bfloat16)
print(a.shape)
b=xetla_kernel.softmax(a,0)