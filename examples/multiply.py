import torch
from baby_triton import triton_jit
from baby_triton.dl_tensor import Tensor


@triton_jit.jit(target="gpu")
def multiply(a: Tensor(shape=(2, 2), dtype="float32"), b: Tensor(shape=(2, 2), dtype="float32")):
    out = a * b
    return out


a = Tensor(shape=(2, 2), dtype="float32")
b = Tensor(shape=(2, 2), dtype="float32")
a.data = torch.ones(size=(2, 2), dtype=torch.float32, device="cuda") * 5
b.data = torch.ones(size=(2, 2), dtype=torch.float32, device="cuda") * 3
print(f"add(): {multiply(a, b)}")
