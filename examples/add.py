import torch
from baby_triton import triton_jit
from baby_triton.dl_tensor import Tensor

# cpu
# @triton_jit.jit(target="cpu")
# def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")):
#     out = a + b
#     return out


# a = Tensor(shape=(2, 3), dtype="float32")
# b = Tensor(shape=(2, 3), dtype="float32")
# a.data = torch.ones(size=(2, 3), dtype=torch.float32)
# b.data = torch.ones(size=(2, 3), dtype=torch.float32)
# print(f"add(): {add(a, b)}")


# gpu
# @triton_jit.jit(target="gpu")
# def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")):
#     out = a + b
#     return out


# gpu example 2
@triton_jit.jit(target="gpu")
def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")):
    out = a + b
    out = out + a
    return out


a = Tensor(shape=(2, 3), dtype="float32")
b = Tensor(shape=(2, 3), dtype="float32")
a.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
b.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
print(f"add(): {add(a, b)}")
